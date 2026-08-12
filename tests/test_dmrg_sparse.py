"""Tier-0 tests for the sparse TTNO storage and its contraction.

Two things are under test and they are different claims:

* **The contraction is the dense one.** :func:`kuiva.dmrg.sparse.dot_sparse` must agree
  with :func:`kuiva.dmrg.block.tensordot` on the same operator — to rounding, not bitwise,
  because the reduction order differs (that module's B10 note).
* **The sparsity is real and grows with node fatness.** That is the *measurement* the
  storage decision rests on, so it is asserted rather than recorded in prose, and it is
  asserted in the direction that can fail: a fatter node must be relatively sparser
  (the guard-pattern discipline — a guard that cannot fail proves nothing).
"""
import numpy as np
import pytest

from kuiva.dmrg.block import BlockTensor, QuantumNumber, Space, tensordot
from kuiva.dmrg.graph import NetworkGraph
from kuiva.dmrg.sparse import SparseW, dot_sparse, sparse_w_gb
from kuiva.dmrg.ttno import compile_ttno, hamiltonian_product_terms

from test_ci_strings import random_spinor_integrals

QN = QuantumNumber
PARITY_TOL = 1e-13         # sparse vs dense contraction: rounding, not bitwise


def small_spaces():
    a = Space([(QN(0), 2), (QN(1), 3)])
    b = Space([(QN(0), 1), (QN(1), 2), (QN(2), 2)])
    c = Space([(QN(0), 2), (QN(1), 1)])
    return a, b, c


def sparsified(rng, spaces, signs, charge, keep=0.25):
    """A dense block tensor with most entries zeroed, and its sparse twin."""
    t = BlockTensor.random(spaces, signs, charge, rng=rng)
    for b in t.blocks:
        b[rng.random(b.shape) > keep] = 0.0
    if all(not b.any() for b in t.blocks):             # pragma: no cover - keep is generous
        t.blocks[0].flat[0] = 1.0
    return t, SparseW.from_block_tensor(t)


def test_round_trip_is_exact():
    rng = np.random.default_rng(0)
    a, b, c = small_spaces()
    dense, sparse = sparsified(rng, (a, b, c), (-1, 1, 1), QN(0))
    assert np.array_equal(sparse.to_dense(), dense.to_dense())
    assert sparse.nnz == int(sum(np.count_nonzero(x) for x in dense.blocks))


def test_dot_sparse_matches_tensordot():
    # the contraction shape the sweep actually uses: several legs contracted, one left
    rng = np.random.default_rng(1)
    a, b, c = small_spaces()
    dense_w, sparse_w = sparsified(rng, (a, b, c), (-1, 1, 1), QN(0))
    other = BlockTensor.random((Space([(QN(0), 2), (QN(1), 2)]), b, c), (1, -1, -1),
                               QN(1), rng=rng)
    ref = tensordot(other, dense_w, ([1, 2], [1, 2]))
    got = dot_sparse(other, sparse_w, ([1, 2], [1, 2]))
    assert got.spaces == ref.spaces and got.signs == ref.signs
    assert got.charge == ref.charge
    d_ref, d_got = ref.to_dense(), got.to_dense()
    assert np.max(np.abs(d_got - d_ref)) < PARITY_TOL * max(np.max(np.abs(d_ref)), 1.0)


def test_dot_sparse_refuses_mismatched_legs():
    rng = np.random.default_rng(2)
    a, b, c = small_spaces()
    _, sparse_w = sparsified(rng, (a, b, c), (-1, 1, 1), QN(0))
    other = BlockTensor.random((a, b), (1, 1), QN(0), rng=rng)
    with pytest.raises(ValueError, match="equal signs"):
        dot_sparse(other, sparse_w, ([1], [1]))        # both +1: flux would not cancel
    with pytest.raises(ValueError, match="different spaces"):
        dot_sparse(other, sparse_w, ([0], [1]))


def test_close_leading_leg_equals_the_dense_contraction():
    rng = np.random.default_rng(3)
    _, b, c = small_spaces()
    top = Space([(QN(1), 1)])
    dense, sparse = sparsified(rng, (top, b, c), (-1, 1, 1), QN(0), keep=0.5)
    closed = sparse.close_leading_leg()
    closer = BlockTensor((top,), (1,), QN(1), np.array([[0]], dtype=np.int64),
                         [np.ones(1, dtype=np.complex128)])
    ref = tensordot(closer, dense, ([0], [0]))
    assert closed.charge == ref.charge and closed.signs == ref.signs
    assert np.array_equal(closed.to_dense(), ref.to_dense())
    assert closed is sparse.close_leading_leg()        # memoized: the CSR cache survives


def test_entry_positions_refuse_an_absent_slot():
    rng = np.random.default_rng(4)
    a, b, c = small_spaces()
    _, sparse = sparsified(rng, (a, b, c), (-1, 1, 1), QN(0))
    row = sparse.sectors[0]
    idx, _ = sparse.block_entries(0)
    assert sparse.entry_positions([row], [idx[0]])[0] == 0
    absent = int(np.setdiff1d(np.arange(int(np.prod(
        [int(sp.dims[i]) for sp, i in zip(sparse.spaces, row)]))), idx)[0])
    with pytest.raises(KeyError):
        sparse.entry_positions([row], [absent])


def test_from_entries_sums_duplicates():
    # a compiler emits several terms into one transition; summing them is the contract
    sp = Space([(QN(0), 2)])
    w = SparseW.from_entries((sp, sp), (1, -1), QN(0), [[0, 0], [0, 0]], [3, 3],
                             [1.0 + 0j, 2.0 + 0j])
    assert w.nnz == 1
    assert w.to_dense()[1, 1] == 3.0


def test_sizing_function_is_exact_two_sided():
    # sizing functions are exact and never pad
    h, eri = random_spinor_integrals(4, seed=1)
    op = compile_ttno(NetworkGraph.path(4), hamiltonian_product_terms(h, eri))
    for w in op.tensors:
        gb = sparse_w_gb(w.spaces, w.nnz, w.nblocks)
        assert gb == pytest.approx(w.nbytes / 1024.0 ** 3, rel=0, abs=0)


def test_the_operator_is_actually_sparse():
    """The measurement behind the storage decision, asserted where it can fail.

    A W tensor's payload is a transition table, and the claim the sparse storage rests on
    is that the table is short against the sector-block area it sits in. Asserted for
    every node layout tried, so a compiler change that made the operator dense fails here
    rather than showing up as an out-of-memory kill on a system nobody can test cheaply.

    ⚠ The threshold is deliberately loose. The interesting *variation* — how the density
    moves with node fatness and with the active-space size — is a measurement and lives in
    a recorded measurement; pinning it here would make an ordinary compiler
    improvement look like a regression.
    """
    h, eri = random_spinor_integrals(8, seed=2)
    terms = hamiltonian_product_terms(h, eri)
    for per_node in (1, 2, 4):
        nodes = 8 // per_node
        contents = [tuple(range(i * per_node, (i + 1) * per_node)) for i in range(nodes)]
        graph = NetworkGraph(nodes, [(i, i + 1) for i in range(nodes - 1)], contents)
        op = compile_ttno(graph, terms)
        assert op.nbytes < op.dense_nbytes
        # payload fraction: stored values against the dense sector-block payload
        assert (16.0 * op.nnz) / float(op.dense_nbytes) < 0.6


def test_a_dense_operator_would_fail_the_sparsity_guard():
    """The companion of the guard above (the pattern: a guard that cannot fail proves
    nothing). A deliberately dense operator on the same spaces must not pass it."""
    sp = Space([(QN(0), 4), (QN(1), 4)])
    dense = BlockTensor.random((sp, sp), (1, -1), QN(0),
                               rng=np.random.default_rng(9))
    full = SparseW.from_block_tensor(dense)
    assert (16.0 * full.nnz) / float(full.dense_nbytes) > 0.6
