"""Tier-0/1 tests for the TTNO compiler.

The oracle is deliberately independent machinery: ``ci.strings.hamiltonian_matrix`` is a
Slater-Condon implementation with its own sign bookkeeping, and ``ci.sigma`` is a third
code path — so agreement here can actually fail. The dense TTNO must equal both, on
every particle-number sector, on paths and on trees, and the model-Hamiltonian seam must
reproduce analytic spin spectra with no integrals involved.
"""
import numpy as np
import pytest

from kuiva.ci.strings import CASSpace, Determinants, hamiltonian_matrix
from kuiva.ci.sigma import SigmaOperator
from kuiva.dmrg.block import QuantumNumber
from kuiva.dmrg.graph import NetworkGraph
from kuiva.dmrg.ttno import (FERMION_MODE, ModeBasis, ProductTerm, compile_ttno,
                             fermion_term, hamiltonian_product_terms)

from test_ci_strings import random_spinor_integrals

QN = QuantumNumber
ORACLE_TOL = 1e-12         # dense TTNO vs an independent Slater-Condon implementation


def kron_index(mask, n_modes):
    """Kron-basis index of a determinant bitmask (mode 0 slowest — the TTNO convention)."""
    idx = 0
    for m in range(n_modes):
        idx = 2 * idx + ((int(mask) >> m) & 1)
    return idx


def sector_indices(n_modes, n_elec):
    """Kron indices of all determinants of one particle-number sector, in mask order."""
    space = CASSpace(n_modes, n_elec, build_map=False)
    return space.masks, np.array([kron_index(m, n_modes) for m in space.masks])


def dense_vs_hamiltonian_matrix(graph, n, seed):
    """Assert the dense TTNO equals ci.strings on every N sector of an n-mode space."""
    h, eri = random_spinor_integrals(n, seed=seed)
    op = compile_ttno(graph, hamiltonian_product_terms(h, eri))
    dense = op.to_dense()
    norm = np.linalg.norm(dense)
    assert np.max(np.abs(dense - dense.conj().T)) < ORACLE_TOL * norm    # Hermitian
    for k in range(n + 1):
        masks, idx = sector_indices(n, k) if k else (np.zeros(1, dtype=np.uint64),
                                                     np.array([0]))
        if k:
            dets = Determinants(masks, n_spinor=n, n_elec=k)
            ref = hamiltonian_matrix(dets, h, eri).toarray()
        else:
            ref = np.zeros((1, 1), dtype=np.complex128)                  # vacuum: E = 0
        block = dense[np.ix_(idx, idx)]
        assert np.max(np.abs(block - ref)) < ORACLE_TOL * max(norm, 1.0), \
            "sector N={} disagrees".format(k)
    return op


def test_ttno_matches_slater_condon_on_a_path():
    dense_vs_hamiltonian_matrix(NetworkGraph.path(5), 5, seed=1)


def test_ttno_matches_slater_condon_on_a_tree():
    #     0 - 1 - 2      a branched 6-mode tree: the same operator, different topology
    #         |
    #         3 - 4
    #         |
    #         5
    graph = NetworkGraph(6, [(0, 1), (1, 2), (1, 3), (3, 4), (3, 5)])
    dense_vs_hamiltonian_matrix(graph, 6, seed=2)


def test_topology_and_root_leave_the_operator_invariant():
    n = 5
    h, eri = random_spinor_integrals(n, seed=3)
    terms = hamiltonian_product_terms(h, eri)
    path = compile_ttno(NetworkGraph.path(n), terms)
    star = compile_ttno(NetworkGraph(n, [(0, i) for i in range(1, n)]), terms)
    rerooted = compile_ttno(NetworkGraph.path(n), terms, root=n // 2)
    ref = path.to_dense()
    assert np.max(np.abs(star.to_dense() - ref)) < ORACLE_TOL * np.linalg.norm(ref)
    assert np.max(np.abs(rerooted.to_dense() - ref)) < ORACLE_TOL * np.linalg.norm(ref)


def test_ttno_application_matches_sigma_on_a_path():
    n, k = 8, 4
    h, eri = random_spinor_integrals(n, seed=4)
    op = compile_ttno(NetworkGraph.path(n), hamiltonian_product_terms(h, eri))
    dense = op.to_dense()
    space = CASSpace(n, k)
    sigma = SigmaOperator(space, h, eri)
    masks, idx = sector_indices(n, k)
    rng = np.random.default_rng(5)
    for _ in range(3):
        c = rng.standard_normal(space.ndet) + 1j * rng.standard_normal(space.ndet)
        full = np.zeros(dense.shape[0], dtype=np.complex128)
        full[idx] = c
        ref = sigma(c)
        got = (dense @ full)[idx]
        assert np.max(np.abs(got - ref)) < 1e-11 * np.linalg.norm(ref)


def test_multi_mode_nodes_give_the_same_operator():
    n = 6
    h, eri = random_spinor_integrals(n, seed=6)
    terms = hamiltonian_product_terms(h, eri)
    single = compile_ttno(NetworkGraph.path(n), terms)
    paired = compile_ttno(NetworkGraph(3, [(0, 1), (1, 2)],
                                      contents=[(0, 1), (2, 3), (4, 5)]), terms)
    ref = single.to_dense()
    assert np.max(np.abs(paired.to_dense() - ref)) < ORACLE_TOL * np.linalg.norm(ref)


def test_an_empty_branching_node_changes_nothing():
    n = 4
    h, eri = random_spinor_integrals(n, seed=7)
    terms = hamiltonian_product_terms(h, eri)
    plain = compile_ttno(NetworkGraph.path(n), terms)
    #   modes 0..3 on the leaves of a star whose hub carries no mode
    hub = NetworkGraph(5, [(4, 0), (4, 1), (4, 2), (4, 3)],
                       contents=[(0,), (1,), (2,), (3,), ()])
    ref = plain.to_dense()
    assert np.max(np.abs(compile_ttno(hub, terms).to_dense() - ref)) \
        < ORACLE_TOL * np.linalg.norm(ref)


def test_operator_bond_dimension_is_compressed():
    """The compression regression guard: the ab initio TTNO must come out O(n^2), not the
    O(n^4) term count. Measured on this construction: max bond dimension 74 / 116 / 173 at
    n = 8 / 10 / 12 — ratios follow n^2 to a few percent, i.e. the complementary-operator
    budget with its parity-dressed variants. The bounds below carry ~2x slack, against a
    per-term regression that would sit at ~n^4/4 across the middle (2500 at n = 10) and
    ~n^3 at an end bond — either failure blows straight through."""
    n = 10
    h, eri = random_spinor_integrals(n, seed=8)
    op = compile_ttno(NetworkGraph.path(n), hamiltonian_product_terms(h, eri))
    dims = op.bond_dimensions()
    assert max(dims.values()) <= 2 * n * n, \
        "max operator bond dimension {} is not O(n^2)".format(max(dims.values()))
    # path rooted at 0: bond (u, u-1) separates subtree {u..n-1} from {0..u-1}
    end_bonds = [d for (u, _), d in dims.items() if min(n - u, u) == 1]
    assert end_bonds and max(end_bonds) <= 2 + 2 * 4, \
        "an end bond carries {} states; a single mode supports at most 8 plus H and 1" \
        .format(max(end_bonds))


# --- the model-Hamiltonian seam (the test seam) ------------------------------------------

SX = np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.complex128)
SY = np.array([[0.0, -0.5j], [0.5j, 0.0]], dtype=np.complex128)
SZ = np.array([[0.5, 0.0], [0.0, -0.5]], dtype=np.complex128)
SPIN_HALF = ModeBasis(2, (QN(0), QN(1)))       # "N" = number of up spins (2Sz shifted)


def heisenberg_terms(edges, j=1.0):
    """Isotropic ``J S_i . S_j`` as S+S-/SzSz products, so the N label is conserved."""
    sp = SX + 1j * SY
    sm = SX - 1j * SY
    terms = []
    for i, k in edges:
        i, k = min(i, k), max(i, k)
        terms.append(ProductTerm(0.5 * j, (i, k), (sp, sm)))
        terms.append(ProductTerm(0.5 * j, (i, k), (sm, sp)))
        terms.append(ProductTerm(1.0 * j, (i, k), (SZ, SZ)))
    return terms


def dense_heisenberg(n, edges, j=1.0):
    """Independent dense construction by plain kron — the oracle for the seam."""
    dim = 2 ** n
    ham = np.zeros((dim, dim), dtype=np.complex128)
    for i, k in edges:
        for s in (SX, SY, SZ):
            factors = [np.eye(2)] * n
            factors[i] = s
            factors[k] = s.copy()
            m = np.eye(1)
            for f in factors:
                m = np.kron(m, f)
            ham += j * m
    return ham


@pytest.mark.parametrize("edges,name", [
    ([(0, 1), (1, 2)], "chain P3"),
    ([(0, 1), (1, 2), (2, 0)], "triangle C3"),
])
def test_model_seam_reproduces_heisenberg(edges, name):
    """No integrals involved: the seam drives the compiler with a known exchange graph.
    The triangle deliberately has an interaction NOT on the tree (edge (2,0) on a path
    graph), so the Z-free generic path is exercised off-topology too."""
    n = 3
    op = compile_ttno(NetworkGraph.path(n), heisenberg_terms(edges), bases=SPIN_HALF)
    dense = op.to_dense()
    ref = dense_heisenberg(n, edges)
    assert np.max(np.abs(dense - ref)) < 1e-13
    # analytic check: eigenvalues of the S=1/2 Heisenberg triangle are -3/4 (4x), +3/4 (4x)
    if name == "triangle C3":
        vals = np.sort(np.linalg.eigvalsh(dense))
        assert np.allclose(vals[:4], -0.75, atol=1e-12)
        assert np.allclose(vals[4:], +0.75, atol=1e-12)


def test_charge_sectors_of_the_model_are_block_diagonal():
    """The N labels must make the TTNO exactly block-diagonal over total Sz sectors."""
    op = compile_ttno(NetworkGraph.path(3), heisenberg_terms([(0, 1), (1, 2)]),
                      bases=SPIN_HALF)
    dense = op.to_dense()
    n_up = np.array([bin(i).count("1") for i in range(8)])
    for a in range(8):
        for b in range(8):
            if n_up[a] != n_up[b]:
                assert dense[a, b] == 0.0


# --- refusals and edge cases --------------------------------------------------------------

def test_terms_with_mismatched_total_charge_are_refused():
    sp = SX + 1j * SY
    terms = [ProductTerm(1.0, (0, 1), (SZ, SZ)), ProductTerm(1.0, (0,), (sp,))]
    with pytest.raises(ValueError, match="charge"):
        compile_ttno(NetworkGraph.path(2), terms, bases=SPIN_HALF)


def test_a_term_on_a_mode_no_node_carries_is_refused():
    terms = [ProductTerm(1.0, (7,), (SZ,))]
    with pytest.raises(ValueError, match="mode"):
        compile_ttno(NetworkGraph.path(2), terms, bases=SPIN_HALF)


def test_fermion_term_conventions():
    # a+_0 a_1 = kron(a+ Z, a) in the JW convention; <{0}| a+_0 a_1 |{1}> = +1
    t = fermion_term(1.0, [(0, True), (1, False)])
    op = compile_ttno(NetworkGraph.path(2), [t])
    dense = op.to_dense()
    assert dense[kron_index(0b01, 2), kron_index(0b10, 2)] == pytest.approx(1.0)
    # a a on one mode vanishes identically
    assert fermion_term(1.0, [(0, False), (0, False)]) is None
    # number operator: a+_p a_p
    tn = fermion_term(1.0, [(1, True), (1, False)])
    assert tn.modes == (1,)
    assert np.array_equal(tn.mats[0], np.diag([0.0, 1.0]).astype(np.complex128))


def test_dense_oracle_refuses_large_spaces():
    h, eri = random_spinor_integrals(4, seed=9)
    op = compile_ttno(NetworkGraph.path(4), hamiltonian_product_terms(h, eri))
    with pytest.raises(ValueError, match="dense"):
        op.to_dense(max_dim=8)


def test_transition_table_sizing_is_exact_on_every_node():
    """Two-sided pin of the compile's own working set against the arrays it builds.

    ⚠ The transition tables are a large allocation the ledger could not see: measured at
    2.2 GB on a 20-spinor five-mode-per-node compile, second only to the two-site
    application's intermediate. They are checked *before* they are built, which needs an
    exact count in advance — so this replays the construction and compares byte for byte,
    on every node of a multi-mode tree where the ``d x d`` local matrices are big enough
    for a mistake to show.
    """
    import kuiva.dmrg.ttno as ttno_mod

    rows = []
    original = ttno_mod._node_tensor

    def recording(u, root, children_u, registries, positions, bond_space, bond_labels,
                  carrying, lab_idx, term_pairs, terms, modes_u, dims_u, phys_u, bases,
                  mats, zero, attach=None, allocations=None):
        d = phys_u[0].total_dim
        id_index = -1 if u == root else registries[u].get(("1",), -1)
        predicted = ttno_mod._transition_table_gb(u, root, children_u, carrying, lab_idx,
                                                  term_pairs, modes_u, d, id_index)
        cache, coefficients, identity_needed = {}, {}, False

        def local_matrix(key):
            mat = cache.get(key)
            if mat is None:
                mat = np.eye(1, dtype=np.complex128)
                for mode, mid in zip(modes_u, key):
                    factor = np.eye(bases[mode].dim, dtype=np.complex128) if mid < 0 \
                        else mats[mid]
                    mat = np.kron(mat, factor)
                cache[key] = mat
            return mat

        for ti, pairs in enumerate(term_pairs):
            out_idx = 0 if u == root else int(lab_idx[ti, u])
            if out_idx == id_index and u != root:
                identity_needed = True
                continue
            in_idx = tuple(int(lab_idx[ti, c]) for c in children_u)
            local = local_matrix(tuple(ttno_mod._term_matid_at(pairs, m) for m in modes_u))
            out_carries = carrying[u][out_idx] if u != root else True
            in_carries = any(bool(carrying[c][i]) for c, i in zip(children_u, in_idx))
            if out_carries and not in_carries and (out_idx, in_idx) not in coefficients:
                coefficients[(out_idx, in_idx)] = terms[ti].coeff * local
        if identity_needed:
            local_matrix(tuple(-1 for _ in modes_u))
        actual = sum(m.nbytes for m in cache.values()) \
            + sum(m.nbytes for m in coefficients.values())
        rows.append((u, predicted, actual))
        return original(u, root, children_u, registries, positions, bond_space,
                        bond_labels, carrying, lab_idx, term_pairs, terms, modes_u,
                        dims_u, phys_u, bases, mats, zero, attach, allocations)

    ttno_mod._node_tensor = recording
    try:
        h, eri = random_spinor_integrals(8, seed=51)
        compile_ttno(NetworkGraph.path(4, contents=[(0, 1), (2, 3), (4, 5), (6, 7)]),
                     hamiltonian_product_terms(h, eri))
    finally:
        ttno_mod._node_tensor = original

    assert len(rows) == 4
    for u, predicted, actual in rows:
        assert predicted * 1024.0 ** 3 == actual, "node {} sizing is not exact".format(u)
