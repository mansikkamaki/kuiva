"""Tier-0 tests for the block-sparse tensor core.

Everything here is exact or near-machine-precision: dense<->block round trips are bitwise
(the embedding moves data, it never does arithmetic), canonical-form identities hold to
rounding, and — the load-bearing part — truncation keeps degenerate groups whole at a tight
AND a deliberately loose threshold, with a companion test asserting that the naive
per-vector cut *does* split the group (the guard-must-be-able-to-fail pattern: a guard that cannot fail
proves nothing).
"""
import numpy as np
import pytest

from kuiva.dmrg.block import (BlockTensor, QuantumNumber, Space, fuse, qr, split, svd)

QN = QuantumNumber

#: A spinor mode: empty or singly occupied.
PHYS = Space([(QN(0), 1), (QN(1), 1)])
#: A generic virtual leg with three particle-number sectors.
VIRT = Space([(QN(0), 2), (QN(1), 3), (QN(2), 2)])

RECON_TOL = 1e-13          # QR/SVD reconstruction and isometry identities, relative


def random_tensor(spaces, signs, charge=None, seed=0):
    return BlockTensor.random(spaces, signs, charge, rng=np.random.default_rng(seed))


def dense_matrix(t, n_left):
    """Dense matricization over the leading ``n_left`` legs."""
    d = t.to_dense()
    return d.reshape(int(np.prod(d.shape[:n_left])), -1)


# --- QuantumNumber ------------------------------------------------------------------------

def test_quantum_number_arithmetic_is_componentwise():
    a, b = QN(2, 1), QN(1, -1)
    assert a + b == QN(3, 0)
    assert a - b == QN(1, 2)
    assert -a == QN(-2, -1)
    assert sum([a, b]) == QN(3, 0)          # __radd__ maps the int 0 start to the identity
    assert a.n == 2 and a.width == 2


def test_quantum_number_width_mismatch_raises():
    with pytest.raises(ValueError, match="width"):
        QN(1) + QN(1, 0)


def test_quantum_number_rejects_non_integers_and_plain_tuples():
    with pytest.raises(TypeError):
        QN(1.5)
    with pytest.raises(TypeError):
        QN(1) + (1,)                        # tuple concatenation must not sneak in
    with pytest.raises(ValueError):
        QN()


def test_quantum_number_orders_lexicographically():
    assert sorted([QN(2), QN(0), QN(1)]) == [QN(0), QN(1), QN(2)]


# --- Space --------------------------------------------------------------------------------

def test_space_sorts_sectors_ascending():
    sp = Space([(QN(2), 2), (QN(0), 1), (QN(1), 3)])
    assert sp.qns == (QN(0), QN(1), QN(2))
    assert list(sp.dims) == [1, 3, 2]
    assert list(sp.offsets) == [0, 1, 4, 6]
    assert sp.total_dim == 6


def test_space_refuses_duplicates_and_bad_dims():
    with pytest.raises(ValueError, match="duplicate"):
        Space([(QN(0), 1), (QN(0), 2)])
    with pytest.raises(ValueError, match="dimension"):
        Space([(QN(0), 0)])
    with pytest.raises(ValueError, match="width"):
        Space([(QN(0), 1), (QN(0, 0), 1)])


# --- block structure ----------------------------------------------------------------------

def test_zeros_enumerates_exactly_the_allowed_blocks():
    t = BlockTensor.zeros((PHYS, PHYS, VIRT), (1, 1, 1), QN(2))
    expected = [(i, j, k) for i in range(2) for j in range(2) for k in range(3)
                if PHYS.qns[i] + PHYS.qns[j] + VIRT.qns[k] == QN(2)]
    assert [tuple(r) for r in t.sectors] == expected
    for row, block in zip(t.sectors, t.blocks):
        assert block.shape == (1, 1, int(VIRT.dims[row[2]]))
        assert block.dtype == np.complex128


def test_dense_round_trip_is_bitwise():
    t = random_tensor((PHYS, VIRT, VIRT), (1, 1, -1), QN(1), seed=3)
    d = t.to_dense()
    t2 = BlockTensor.from_dense(d, t.spaces, t.signs, t.charge)
    assert np.array_equal(t2.sectors, t.sectors)
    for b1, b2 in zip(t.blocks, t2.blocks):
        assert np.array_equal(b1, b2)                    # bitwise, not allclose
    assert np.array_equal(t2.to_dense(), d)


def test_dense_forbidden_entries_are_exactly_zero():
    t = random_tensor((PHYS, PHYS, VIRT), (1, 1, 1), QN(2), seed=4)
    d = t.to_dense()
    for i in range(2):
        for j in range(2):
            for k in range(3):
                if PHYS.qns[i] + PHYS.qns[j] + VIRT.qns[k] != QN(2):
                    sl = (slice(i, i + 1), slice(j, j + 1),
                          slice(int(VIRT.offsets[k]), int(VIRT.offsets[k + 1])))
                    assert np.all(d[sl] == 0.0)


def test_from_dense_refuses_symmetry_broken_input():
    t = random_tensor((PHYS, VIRT), (1, -1), QN(0), seed=5)
    d = t.to_dense()
    d[1, 0] += 0.1                       # (N=1) row against an (N=0) column: forbidden
    with pytest.raises(ValueError, match="forbidden"):
        BlockTensor.from_dense(d, t.spaces, t.signs, t.charge)


def test_norm_transpose_conj_match_dense():
    t = random_tensor((PHYS, VIRT, PHYS), (1, -1, 1), QN(0), seed=6)
    d = t.to_dense()
    assert t.norm() == pytest.approx(np.linalg.norm(d), rel=1e-14)
    assert np.array_equal(t.transpose((2, 0, 1)).to_dense(), d.transpose((2, 0, 1)))
    tc = t.conj()
    assert np.array_equal(tc.to_dense(), d.conj())
    assert tc.signs == tuple(-s for s in t.signs)
    assert tc.charge == -t.charge


def test_find_returns_the_block_or_none():
    t = random_tensor((PHYS, PHYS), (1, -1), QN(0), seed=7)
    present = tuple(int(i) for i in t.sectors[0])
    assert t.find(present) is t.blocks[0]
    assert t.find((0, 1)) is None                        # N flux 0 - 1 != 0: never a block


# --- fuse / split -------------------------------------------------------------------------

def test_fuse_split_round_trip_is_exact():
    t = random_tensor((PHYS, PHYS, VIRT), (1, 1, 1), QN(2), seed=8)
    for axes in [(0, 1), (1, 2), (0, 2), (0, 1, 2)]:
        fused, record = fuse(t, axes)
        assert fused.ndim == t.ndim - len(axes) + 1
        assert fused.norm() == pytest.approx(t.norm(), rel=1e-15)
        back = split(fused, record, axis=0)
        # split re-expands the legs at position 0 in fused order; restore the original order
        rest = [a for a in range(t.ndim) if a not in axes]
        inverse = np.argsort(list(axes) + rest)
        assert np.array_equal(back.transpose(inverse).to_dense(), t.to_dense())


def test_split_refuses_a_mismatched_record():
    t = random_tensor((PHYS, PHYS, VIRT), (1, 1, 1), QN(2), seed=9)
    _, record = fuse(t, (0, 1))
    other, _ = fuse(t, (1, 2))
    with pytest.raises(ValueError, match="does not match"):
        split(other, record, axis=0)


# --- QR -----------------------------------------------------------------------------------

def test_qr_reconstructs_and_q_is_an_isometry():
    t = random_tensor((PHYS, PHYS, VIRT), (1, 1, 1), QN(2), seed=10)
    q, r = qr(t, (0, 1))
    assert q.charge == QN(0) and r.charge == t.charge
    qd = dense_matrix(q, 2)                              # (left-flat, bond)
    rd = dense_matrix(r, 1)                              # (bond, right-flat)
    ident = qd.conj().T @ qd
    assert np.max(np.abs(ident - np.eye(ident.shape[0]))) < RECON_TOL
    td = dense_matrix(t, 2)
    assert np.max(np.abs(qd @ rd - td)) < RECON_TOL * np.linalg.norm(td)


def test_qr_is_deterministic_and_diag_r_is_real_nonnegative():
    # single-sector spaces make the dense R literally upper triangular, so the phase
    # convention (diag R real >= 0) is directly inspectable
    left = Space([(QN(0), 4)])
    right = Space([(QN(0), 3)])
    t = random_tensor((left, right), (1, 1), QN(0), seed=11)
    q1, r1 = qr(t, (0,))
    q2, r2 = qr(t, (0,))
    for b1, b2 in zip(q1.blocks + r1.blocks, q2.blocks + r2.blocks):
        assert np.array_equal(b1, b2)                    # bitwise reproducible
    diag = np.diagonal(r1.to_dense())
    assert np.all(np.abs(diag.imag) < 1e-14)
    assert np.all(diag.real > -1e-14)


def test_qr_rejects_improper_bipartitions():
    t = random_tensor((PHYS, VIRT), (1, -1), QN(0), seed=12)
    for bad in [(), (0, 1), (0, 0), (5,)]:
        with pytest.raises(ValueError):
            qr(t, bad)


# --- SVD and the degenerate-group truncation ----------------------------------------------

def prescribed_spectrum_tensor(sector_values, seed=20):
    """A tensor whose Schmidt spectrum across the (0, 1) | (2,) cut is prescribed.

    ``sector_values`` maps bond sector index -> descending values. The bond has three
    sectors (left flux N = 0, 1, 2 with dims 1, 2, 1), so degenerate values can be placed
    either inside one sector or straddling two — the case per-sector truncation cannot see.
    """
    spaces = (PHYS, PHYS, Space([(QN(0), 2), (QN(1), 2), (QN(2), 2)]))
    signs = (1, 1, 1)
    charge = QN(2)
    t0 = random_tensor(spaces, signs, charge, seed=seed)
    u, s_sec, vh, info = svd(t0, (0, 1))
    assert info.bond_dim == 4                            # control: nothing was truncated
    flat = np.concatenate([np.asarray(sector_values[k], dtype=np.float64)
                           for k in range(len(s_sec))])
    ud = dense_matrix(u, 2)
    vhd = dense_matrix(vh, 1)
    td = (ud @ np.diag(flat) @ vhd).reshape(t0.to_dense().shape)
    return BlockTensor.from_dense(td, spaces, signs, charge), spaces


def test_svd_reconstructs_without_truncation():
    t = random_tensor((PHYS, PHYS, VIRT), (1, 1, 1), QN(2), seed=13)
    u, s_sec, vh, info = svd(t, (0, 1))
    assert info.discarded_weight == 0.0 and info.n_discarded == 0
    flat = np.concatenate([s for s in s_sec])
    ud, vhd, td = dense_matrix(u, 2), dense_matrix(vh, 1), dense_matrix(t, 2)
    assert np.max(np.abs(ud @ np.diag(flat) @ vhd - td)) < RECON_TOL * np.linalg.norm(td)
    ident = ud.conj().T @ ud
    assert np.max(np.abs(ident - np.eye(ident.shape[0]))) < RECON_TOL
    for s in s_sec:
        assert np.all(np.diff(s) <= 0)                   # descending within each sector


def test_truncation_keeps_a_degenerate_group_whole_across_sectors():
    """The load-bearing test: a degenerate pair straddles two bond sectors, and a cap that
    cannot hold both must keep neither — not whichever one a per-vector rule would pick."""
    t, _ = prescribed_spectrum_tensor({0: [1.0], 1: [0.6, 0.3], 2: [0.6]})
    # cap the bond at 2: group {0.6, 0.6} needs kept = 3, so it is not started
    u, s_sec, vh, info = svd(t, (0, 1), max_bond=2)
    assert info.bond_dim == 1
    kept = np.concatenate([s for s in s_sec])
    assert np.sum(np.abs(kept - 0.6) < 1e-9) == 0        # both members out, never one
    # with room for the whole group, both members come in together
    u, s_sec, vh, info = svd(t, (0, 1), max_bond=3)
    assert info.bond_dim == 3
    kept = np.concatenate([s for s in s_sec])
    assert np.sum(np.abs(kept - 0.6) < 1e-9) == 2


def test_companion_the_naive_per_vector_cut_does_split_the_group():
    """The guard pattern: assert the naive rule commits exactly the error the
    group-complete rule exists to prevent — otherwise the test above proves nothing."""
    t, _ = prescribed_spectrum_tensor({0: [1.0], 1: [0.6, 0.3], 2: [0.6]})
    u, s_sec, vh, info = svd(t, (0, 1))                  # untruncated spectrum
    flat = np.sort(np.concatenate([s for s in s_sec]))[::-1]
    naive_kept = flat[:2]                                # plain top-k at max_bond = 2
    in_group = np.abs(naive_kept - 0.6) < 1e-9
    assert np.sum(in_group) == 1                         # the naive cut splits the pair


def test_truncation_by_tolerance_drops_whole_groups_tight_and_loose():
    t, _ = prescribed_spectrum_tensor({0: [1.0], 1: [0.6, 0.3], 2: [0.6]})
    # tight: only the 0.3 tail goes
    u, s_sec, vh, info = svd(t, (0, 1), tol=0.45)
    assert info.bond_dim == 3
    assert info.discarded_weight == pytest.approx(0.3 ** 2 / (1 + 2 * 0.36 + 0.09), rel=1e-9)
    # deliberately loose, landing above the degenerate pair: the group goes whole
    u, s_sec, vh, info = svd(t, (0, 1), tol=0.75)
    assert info.bond_dim == 1
    kept = np.concatenate([s for s in s_sec])
    assert np.sum(np.abs(kept - 0.6) < 1e-9) == 0


def test_truncated_reconstruction_error_equals_the_discarded_weight():
    t, _ = prescribed_spectrum_tensor({0: [1.0], 1: [0.6, 0.3], 2: [0.6]})
    u, s_sec, vh, info = svd(t, (0, 1), tol=0.5)
    flat = np.concatenate([s for s in s_sec])
    approx = dense_matrix(u, 2) @ np.diag(flat) @ dense_matrix(vh, 1)
    err2 = np.linalg.norm(dense_matrix(t, 2) - approx) ** 2
    assert err2 == pytest.approx(info.discarded_weight * t.norm() ** 2, rel=1e-9)


def test_stability_floor_drops_noise_groups_whole():
    t, _ = prescribed_spectrum_tensor({0: [1.0], 1: [1e-15, 1e-16], 2: [0.5]})
    u, s_sec, vh, info = svd(t, (0, 1))
    assert info.bond_dim == 2                            # the noise tail went whole
    assert info.smallest_kept == pytest.approx(0.5, rel=1e-9)


def test_a_cap_inside_the_leading_degenerate_group_is_refused_not_rounded():
    t, _ = prescribed_spectrum_tensor({0: [0.8], 1: [1.0, 0.4], 2: [1.0]})
    with pytest.raises(ValueError, match="refused"):
        svd(t, (0, 1), max_bond=1)


def test_svd_of_a_zero_tensor_is_refused():
    t = BlockTensor.zeros((PHYS, VIRT), (1, -1), QN(0))
    with pytest.raises(ValueError, match="zero"):
        svd(t, (0,))


def test_nonleading_bipartition_reconstructs():
    """left_axes need not be the leading legs; check via an explicit transpose."""
    t = random_tensor((PHYS, VIRT, PHYS), (1, -1, 1), QN(0), seed=14)
    u, s_sec, vh, info = svd(t, (0, 2))
    flat = np.concatenate([s for s in s_sec])
    td = t.to_dense().transpose((0, 2, 1)).reshape(4, -1)
    approx = dense_matrix(u, 2) @ np.diag(flat) @ dense_matrix(vh, 1)
    assert np.max(np.abs(approx - td)) < RECON_TOL * np.linalg.norm(td)
