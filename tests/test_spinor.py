"""Tier-0 tests for the scalar -> spinor expansion.

Everything here is exact by construction: T^2 = -1, Kramers pairing, orthonormality, and the
index algebra of the interleaved ordering. "T^2 = -1 on odd-electron states" is a
Tier-0 invariance check; this file establishes it at the one-particle level, where a violation
is unambiguous and cheap to find.
"""
import numpy as np
import pytest

from kuiva.spinor.expand import (SpinorBasis, barred, expand_scalar_mos, is_barred,
                                 is_time_reversal_even, kramers_block_permutation,
                                 spatial_index, spin_block_diagonal, spinor_indices,
                                 time_reverse, two_component_operator, unbarred)


def random_mos(nbas=8, nmo=8, seed=0):
    """Orthonormal real 'scalar MOs' in an orthonormal scalar basis."""
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((nbas, nbas)))
    return q[:, :nmo]


# --- Time reversal ----------------------------------------------------------------------
def test_time_reversal_squares_to_minus_one():
    """The origin of Kramers degeneracy; fixes the sign convention of T = -i sigma_y K."""
    rng = np.random.default_rng(1)
    c = rng.standard_normal((10, 4)) + 1j * rng.standard_normal((10, 4))
    assert np.max(np.abs(time_reverse(time_reverse(c)) + c)) < 1e-14


def test_time_reversal_is_antiunitary():
    """<Tx|Ty> = <y|x>: the inner product is conjugated, norms are preserved."""
    rng = np.random.default_rng(2)
    x = rng.standard_normal((8, 1)) + 1j * rng.standard_normal((8, 1))
    y = rng.standard_normal((8, 1)) + 1j * rng.standard_normal((8, 1))
    lhs = (time_reverse(x).conj().T @ time_reverse(y))[0, 0]
    rhs = (y.conj().T @ x)[0, 0]
    assert abs(lhs - rhs) < 1e-14


def test_a_spinor_is_orthogonal_to_its_kramers_partner():
    rng = np.random.default_rng(3)
    c = rng.standard_normal((12, 1)) + 1j * rng.standard_normal((12, 1))
    c /= np.linalg.norm(c)
    assert abs((c.conj().T @ time_reverse(c))[0, 0]) < 1e-14


def test_odd_rows_rejected():
    with pytest.raises(ValueError, match="even number of rows"):
        time_reverse(np.zeros((7, 2)))


# --- The expansion ----------------------------------------------------------------------
def test_expansion_shape_and_pairing():
    mo = random_mos(9, 7, seed=4)
    sb = expand_scalar_mos(mo, np.arange(7.0), np.array([2.0] * 3 + [0.0] * 4))
    assert sb.nbas == 9 and sb.nspinor == 14
    assert sb.partner_deviation() < 1e-14            # exactly Kramers paired by construction
    assert sb.orthonormality_error() < 1e-12
    assert np.allclose(sb.energy[0::2], sb.energy[1::2])
    assert np.allclose(sb.occ[:6], 1.0) and np.allclose(sb.occ[6:], 0.0)


def test_expansion_layout_is_the_documented_one():
    """Rows blocked [alpha|beta]; columns interleaved (p, pbar). CI addressing depends on it."""
    mo = random_mos(5, 5, seed=5)
    sb = expand_scalar_mos(mo)
    nb = sb.nbas
    for p in range(5):
        assert np.allclose(sb.c[:nb, unbarred(p)], mo[:, p])     # (phi, 0)
        assert np.allclose(sb.c[nb:, unbarred(p)], 0.0)
        assert np.allclose(sb.c[:nb, barred(p)], 0.0)            # (0, phi)
        assert np.allclose(sb.c[nb:, barred(p)], mo[:, p])
    assert np.allclose(sb.c[:, 1::2], time_reverse(sb.c[:, 0::2]))


def test_orthonormality_with_a_metric():
    """The expansion is basis-agnostic: it works on AO coefficients with an overlap too."""
    rng = np.random.default_rng(6)
    a = rng.standard_normal((6, 6))
    s = a.T @ a
    w, v = np.linalg.eigh(s)
    mo = v / np.sqrt(w)                               # S-orthonormal columns
    sb = expand_scalar_mos(mo, basis="ao")
    assert sb.orthonormality_error(s) < 1e-10


def test_index_algebra():
    assert unbarred(3) == 6 and barred(3) == 7
    assert spatial_index(6) == 3 and spatial_index(7) == 3
    assert not is_barred(6) and is_barred(7)
    assert list(spinor_indices([1, 4])) == [2, 3, 8, 9]     # both partners, never split
    perm = kramers_block_permutation(8)
    assert list(perm) == [0, 2, 4, 6, 1, 3, 5, 7]
    with pytest.raises(ValueError):
        kramers_block_permutation(7)


def test_active_space_slicing_is_contiguous():
    """A scalar active space must map to a contiguous spinor range (the reason for the
    interleaved ordering)."""
    idx = spinor_indices(range(4, 9))
    assert list(idx) == list(range(8, 18))


def test_basis_change_round_trip():
    """Re-expressing the spinors over another scalar basis and back is the identity."""
    mo = random_mos(7, 7, seed=7)
    sb = expand_scalar_mos(mo)
    q, _ = np.linalg.qr(np.random.default_rng(8).standard_normal((7, 7)))
    moved = sb.transform_scalar_basis(q, basis="ao")
    back = moved.transform_scalar_basis(q.T, basis="working")
    assert np.max(np.abs(back.c - sb.c)) < 1e-12
    assert moved.partner_deviation() < 1e-14           # a real basis change preserves pairing


# --- Two-component operators -------------------------------------------------------------
def test_spin_free_operator_is_block_diagonal_and_time_even():
    rng = np.random.default_rng(9)
    a = rng.standard_normal((5, 5))
    a = a + a.T
    big = spin_block_diagonal(a)
    assert np.allclose(big[:5, 5:], 0.0) and np.allclose(big[5:, :5], 0.0)
    assert is_time_reversal_even(big)


def test_spin_orbit_operator_is_hermitian_and_time_even():
    """sigma.W is time-reversal EVEN (spin and orbital angular momentum are both time-odd).

    This is the structural check that catches a swapped or transposed spin block in the
    assembly — an error no hermiticity or norm test can see.
    """
    rng = np.random.default_rng(10)
    a = rng.standard_normal((6, 6))
    a = a + a.T
    w = rng.standard_normal((3, 6, 6))
    w = w - np.transpose(w, (0, 2, 1))                 # real antisymmetric, as SO integrals are
    h = two_component_operator(a, w)
    assert np.max(np.abs(h - h.conj().T)) < 1e-12      # Hermitian
    assert is_time_reversal_even(h)
    assert not np.allclose(h[:6, 6:], 0.0)             # genuinely spin-coupling


def test_time_reversal_odd_operator_is_detected():
    """A deliberately broken assembly (Sz-like, time-odd) must fail the check."""
    n = 4
    h = np.zeros((2 * n, 2 * n), dtype=complex)
    h[:n, :n] = np.eye(n)
    h[n:, n:] = -np.eye(n)                             # ~ sigma_z, time-odd
    assert not is_time_reversal_even(h)


def test_spin_orbit_factors_validated():
    a = np.eye(4)
    with pytest.raises(ValueError, match="shape"):
        two_component_operator(a, np.zeros((2, 4, 4)))


def test_non_antisymmetric_so_factors_warn(kuiva_caplog):
    a = np.eye(4)
    w = np.zeros((3, 4, 4))
    w[0, 0, 1] = w[0, 1, 0] = 1.0                      # symmetric: not a valid SO factor
    with kuiva_caplog.at_level("WARNING"):
        two_component_operator(a, w)
    assert any("antisymmetric" in r.message for r in kuiva_caplog.records)


def test_antisymmetry_check_is_relative_not_absolute(kuiva_caplog):
    """⚠ The threshold must scale with the operator, and it did not.

    :func:`two_component_operator` is called on things spanning ten orders of magnitude: an
    atomic mean-field correction at ~1e-3 Eh and the raw ``W = <sigma.p V sigma.p>`` integrals
    of a heavy element at ~1e+6 Eh. The check was absolute at 1e-10, which is wrong at **both**
    ends, and this test pins both:

    * a large operator whose antisymmetry is exact to rounding must **not** warn — the old
      threshold fired on every heavy element (TlH: 9.3e-10 absolute, 1.5e-16 relative);
    * a small operator with a genuinely broken 1e-8 *relative* asymmetry **must** warn — the
      old threshold passed it silently, which is the more dangerous half.
    """
    rng = np.random.default_rng(0)
    n = 6
    base = rng.standard_normal((3, n, n))
    anti = base - np.transpose(base, (0, 2, 1))

    # Large, and antisymmetric to machine precision: silence.
    big = anti * 1e6
    with kuiva_caplog.at_level("WARNING"):
        two_component_operator(np.eye(n), big)
    assert not [r for r in kuiva_caplog.records if "antisymmetric" in r.message]

    # Small, and broken at 1e-8 relative: a warning.
    small = anti * 1e-3
    small[0, 0, 1] += 1e-8 * float(np.max(np.abs(small)))
    with kuiva_caplog.at_level("WARNING"):
        two_component_operator(np.eye(n), small)
    assert [r for r in kuiva_caplog.records if "antisymmetric" in r.message]
