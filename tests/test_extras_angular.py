"""The angular algebra of the atomic Slater-Condon parameters, against closed forms.

Every number in :mod:`kuiva.extras.angular` is determined by angular momenta alone, so all of
it can be checked against something that is *not* a rerun of the same code. Four independent
authorities are used, and which one is used where is the point of this file:

1. **Published closed forms** — the Racah expression's own special cases, the ``c^k`` tables of
   Condon & Shortley and Cowan, the ``l . s`` spectrum. These catch a wrong formula.
2. **Numerical quadrature on the sphere.** The ``A^k`` tensors are rebuilt from the Laplace
   expansion by integrating products of harmonics on a Gauss-Legendre times uniform grid,
   evaluating the harmonics from associated Legendre functions. This shares no line of code
   with the 3j route and is what would catch a wrong index placement, a missing conservation
   delta or a dropped conjugation — the errors that leave a plausible, symmetric tensor.
3. **The integral library itself.** The complex-to-real transform is checked by evaluating
   real basis functions on a sphere and confirming that each shell is one constant times the
   module's own real harmonics. ⚠ This is the only check of the *sign* convention, and a sign
   is not a detail: it flips every parameter with an odd number of indices in that channel.
4. **Algebraic identities the tensors must satisfy** — permutational symmetry, the closed-shell
   trace rule, the angular-momentum commutators.

Nothing here touches an SCF or an integral file; the whole file is well under a second bar the
one PySCF ``Mole`` construction of check 3.
"""
import math

import numpy as np
import pytest
from scipy.special import lpmv

from kuiva.extras.angular import (admissible_k, angular_momentum_matrices, angular_tensor,
                                  complex_to_real, condon_shortley_ck, couples, gaunt,
                                  spin_orbit_matrix, wigner_3j)


# --- 1. Published closed forms --------------------------------------------------------------

def test_the_3j_symbols_match_their_closed_forms():
    """``(j1 j2 j3; 0 0 0)^2`` for the couplings the ``F^k`` of s, p, d and f shells need.

    These are the values Cowan's configuration-average coefficients are built from, and they
    are the ones a floating-point Racah sum starts losing digits on first.
    """
    assert wigner_3j(1, 2, 1, 0, 0, 0) ** 2 == pytest.approx(2 / 15)
    assert wigner_3j(2, 2, 2, 0, 0, 0) ** 2 == pytest.approx(2 / 35)
    assert wigner_3j(2, 4, 2, 0, 0, 0) ** 2 == pytest.approx(2 / 35)
    assert wigner_3j(3, 2, 3, 0, 0, 0) ** 2 == pytest.approx(4 / 105)
    assert wigner_3j(3, 4, 3, 0, 0, 0) ** 2 == pytest.approx(2 / 77)
    assert wigner_3j(3, 6, 3, 0, 0, 0) ** 2 == pytest.approx(100 / 3003)
    # j3 = 0 collapses to a closed form for every m, and an odd sum of j vanishes identically
    for j in range(5):
        for m in range(-j, j + 1):
            assert wigner_3j(j, j, 0, m, -m, 0) == pytest.approx(
                (-1) ** (j - m) / math.sqrt(2 * j + 1))
    assert wigner_3j(1, 1, 1, 0, 0, 0) == 0.0       # odd sum of l, killed by parity
    assert wigner_3j(2, 3, 4, 0, 0, 0) == 0.0       # a valid triangle, still odd


def test_the_3j_symbols_obey_orthogonality_and_the_permutation_phases():
    """Summed over all magnetic quantum numbers a 3j symbol is normalized to one, and column
    permutations and an overall sign change of the ``m`` are phases rather than new numbers.
    Between them these pin down the normalization and the phase convention."""
    for (j1, j2, j3) in [(1, 2, 1), (3, 4, 3), (2, 2, 2), (3, 3, 4)]:
        total = sum(wigner_3j(j1, j2, j3, m1, m2, -m1 - m2) ** 2
                    for m1 in range(-j1, j1 + 1) for m2 in range(-j2, j2 + 1))
        assert total == pytest.approx(1.0)

    rng = np.random.default_rng(11)
    for _ in range(40):
        j1, j2, j3 = (int(v) for v in rng.integers(0, 5, size=3))
        m1, m2 = int(rng.integers(-j1, j1 + 1)), int(rng.integers(-j2, j2 + 1))
        m3 = -m1 - m2
        value = wigner_3j(j1, j2, j3, m1, m2, m3)
        phase = (-1) ** (j1 + j2 + j3)
        assert wigner_3j(j2, j3, j1, m2, m3, m1) == pytest.approx(value)      # cyclic
        assert wigner_3j(j2, j1, j3, m2, m1, m3) == pytest.approx(phase * value)
        assert wigner_3j(j1, j2, j3, -m1, -m2, -m3) == pytest.approx(phase * value)


def test_half_integer_angular_momenta_are_refused():
    """Spin never enters this module's arithmetic, so a half-integer argument is a caller
    error rather than a case to support quietly."""
    with pytest.raises(ValueError, match="integer"):
        wigner_3j(0.5, 0.5, 1, 0.5, -0.5, 0)


def test_the_condon_shortley_coefficients_match_the_published_tables():
    """``c^k(lm; l'm')`` against the tabulated values (Condon & Shortley 1935; Cowan 1981).

    ``c^0(lm; lm) = 1`` fixes the normalization, and the ``k > 0`` entries fix the phase: the
    ``c^2(1,1; 1,1) = -1/5`` sign is the one a missing Condon-Shortley factor would flip.
    """
    for l in range(4):
        for m in range(-l, l + 1):
            assert condon_shortley_ck(0, l, m, l, m) == pytest.approx(1.0)
    assert condon_shortley_ck(2, 1, 1, 1, 1) == pytest.approx(-1 / 5)
    assert condon_shortley_ck(2, 1, 0, 1, 0) == pytest.approx(2 / 5)
    assert condon_shortley_ck(2, 1, 1, 1, -1) == pytest.approx(-math.sqrt(6) / 5)
    assert condon_shortley_ck(2, 2, 0, 2, 0) == pytest.approx(2 / 7)
    assert condon_shortley_ck(4, 2, 0, 2, 0) == pytest.approx(2 / 7)
    assert condon_shortley_ck(2, 3, 0, 3, 0) == pytest.approx(4 / 15)
    assert condon_shortley_ck(4, 3, 0, 3, 0) == pytest.approx(2 / 11)
    assert condon_shortley_ck(6, 3, 0, 3, 0) == pytest.approx(100 / 429)


def test_the_selection_rules_are_the_triangle_and_the_parity():
    """The two rules together, on the case the whole feature was specified around: the
    chemists' integral ``(4f 4f | 6s 5d)`` is nonzero and has exactly one ``k``."""
    assert admissible_k(3, 3, 0, 2) == (2,)                  # (4f 4f | 6s 5d)
    assert admissible_k(3, 3, 3, 3) == (0, 2, 4, 6)          # F^k(4f, 4f)
    assert admissible_k(3, 0, 0, 3) == (3,)                  # G^3(4f, 6s)
    assert admissible_k(3, 2, 2, 3) == (1, 3, 5)             # G^k(4f, 5d)
    assert admissible_k(0, 0, 3, 3) == (0,)                  # F^0(6s, 4f)
    assert admissible_k(3, 2, 0, 0) == ()                    # (4f 5d | 6s 6s): no k at all
    assert couples(1, 1, 0) and couples(1, 1, 2)
    assert not couples(1, 1, 1) and not couples(1, 1, 4)


def test_the_spin_orbit_matrix_has_the_two_levels_of_a_shell():
    """``l . s`` splits a shell into ``j = l +- 1/2`` with eigenvalues ``l/2`` and
    ``-(l+1)/2`` and multiplicities ``2l+2`` and ``2l``. Nothing about the orbital basis can
    change that, so it is the closed form the fit of a spin-orbit constant is judged against.
    """
    for l in range(1, 5):
        eigenvalues = np.linalg.eigvalsh(spin_orbit_matrix(l))
        lower = eigenvalues[eigenvalues < 0.0]
        upper = eigenvalues[eigenvalues > 0.0]
        assert lower.size == 2 * l and upper.size == 2 * l + 2
        assert np.allclose(lower, -(l + 1) / 2.0)
        assert np.allclose(upper, l / 2.0)
        # Hermitian, traceless, and the splitting is (2l+1)/2 times the constant
        ls = spin_orbit_matrix(l)
        assert np.allclose(ls, ls.conj().T)
        assert abs(np.trace(ls)) < 1e-12
        assert upper[0] - lower[0] == pytest.approx((2 * l + 1) / 2.0)


def test_the_angular_momentum_matrices_close_the_algebra():
    """``[Lx, Ly] = i Lz`` and ``L^2 = l(l+1)`` in the real basis — the statement that the
    complex-to-real rotation was applied as ``U^* O U^T`` and not as ``U O U^dag``, which is
    the one transformation error that survives every hermiticity check."""
    for l in range(1, 5):
        lx, ly, lz = angular_momentum_matrices(l)
        n = 2 * l + 1
        assert np.allclose(lx @ ly - ly @ lx, 1j * lz)
        assert np.allclose(ly @ lz - lz @ ly, 1j * lx)
        assert np.allclose(lx @ lx + ly @ ly + lz @ lz, l * (l + 1) * np.eye(n))
        for op in (lx, ly, lz):
            assert np.allclose(op, op.conj().T)
            assert np.allclose(op.real, 0.0)          # purely imaginary in a real basis


# --- 2. The A^k tensors against numerical quadrature on the sphere ---------------------------

def _complex_harmonic(l, m, cos_theta, phi):
    """``Y_l^m`` with the Condon-Shortley phase, from associated Legendre functions.

    Written out here rather than imported so that the quadrature check below shares nothing
    with the module under test except the definition of a spherical harmonic itself.
    ``scipy.special.lpmv`` already carries the Condon-Shortley factor.
    """
    am = abs(m)
    norm = math.sqrt((2 * l + 1) / (4 * math.pi)
                     * math.factorial(l - am) / math.factorial(l + am))
    y = norm * lpmv(am, l, cos_theta) * np.exp(1j * am * phi)
    return y if m >= 0 else (-1) ** am * np.conj(y)


def _sphere_grid(n_theta=48, n_phi=64):
    """Gauss-Legendre in ``cos(theta)`` times a uniform ``phi`` grid, with the weights.

    Exact for the integrands here: a product of harmonics is a polynomial in ``cos(theta)``
    (the powers of ``sin(theta)`` always pair up) times a trigonometric polynomial in ``phi``,
    both far below the resolution of this grid.
    """
    nodes, weights = np.polynomial.legendre.leggauss(n_theta)
    phi = 2 * math.pi * np.arange(n_phi) / n_phi
    cos_theta = nodes[:, None] * np.ones(n_phi)[None, :]
    phi_grid = np.ones(n_theta)[:, None] * phi[None, :]
    w = weights[:, None] * np.full(n_phi, 2 * math.pi / n_phi)[None, :]
    return cos_theta.ravel(), phi_grid.ravel(), w.ravel()


def _real_harmonics(l, cos_theta, phi):
    """The module's real harmonics on the grid: ``U`` applied to complex ones. ``(2l+1, npt)``."""
    complex_set = np.array([_complex_harmonic(l, mu, cos_theta, phi)
                            for mu in range(-l, l + 1)])
    return complex_to_real(l) @ complex_set


def _angular_tensor_by_quadrature(l_p, l_q, l_r, l_s, k):
    """``A^k`` rebuilt from the Laplace expansion of ``1/r_12``, by integrating on a grid.

    .. math::

        A^k = \\frac{4\\pi}{2k+1} \\sum_q
              \\Big[\\int \\phi_p Y_{kq}^* \\phi_q\\Big]
              \\Big[\\int \\phi_r Y_{kq} \\phi_s\\Big]

    for real orbitals ``phi``. The sum over ``q`` is explicit here, where the module's route
    collapses it analytically with a conservation delta — which is exactly the step this
    check exists to test.
    """
    cos_theta, phi, w = _sphere_grid()
    yp, yq = _real_harmonics(l_p, cos_theta, phi), _real_harmonics(l_q, cos_theta, phi)
    yr, ys = _real_harmonics(l_r, cos_theta, phi), _real_harmonics(l_s, cos_theta, phi)
    total = np.zeros((2 * l_p + 1, 2 * l_q + 1, 2 * l_r + 1, 2 * l_s + 1), dtype=complex)
    for q in range(-k, k + 1):
        ykq = _complex_harmonic(k, q, cos_theta, phi)
        one = np.einsum("ax,bx,x->ab", yp, yq, np.conj(ykq) * w)
        two = np.einsum("cx,dx,x->cd", yr, ys, ykq * w)
        total += one[:, :, None, None] * two[None, None, :, :]
    return (4 * math.pi / (2 * k + 1)) * total


@pytest.mark.parametrize("shells", [(1, 1, 1, 1), (3, 3, 3, 3), (3, 3, 0, 2), (2, 1, 1, 2),
                                    (3, 2, 2, 3), (0, 2, 3, 3), (2, 2, 1, 1)])
def test_the_angular_tensors_agree_with_quadrature_on_the_sphere(shells):
    """⚠ **The check that can actually fail on a wrong tensor.**

    The module builds ``A^k`` from exact 3j symbols, two Condon-Shortley coefficients and a
    conservation delta, then rotates four indices to real harmonics with a conjugation on two
    of them. Every one of those steps has a plausible wrong version — indices transposed, the
    delta omitted, the conjugation on the wrong pair — and each produces a tensor that is
    still real, still symmetric under the obvious swaps, and wrong. Integrating the Laplace
    expansion directly on a grid shares none of that machinery.
    """
    l_p, l_q, l_r, l_s = shells
    ks = admissible_k(l_p, l_q, l_r, l_s)
    assert ks, "the parametrization should only carry classes that survive the selection rules"
    for k in ks:
        mine = angular_tensor(l_p, l_q, l_r, l_s, k)
        theirs = _angular_tensor_by_quadrature(l_p, l_q, l_r, l_s, k)
        assert np.max(np.abs(theirs.imag)) < 1e-10
        assert np.allclose(mine, theirs.real, atol=1e-10)
    # and a forbidden k really is zero, rather than merely not enumerated
    forbidden = [k for k in range(0, l_p + l_q + l_r + l_s + 1) if k not in ks]
    for k in forbidden[:3]:
        assert np.max(np.abs(_angular_tensor_by_quadrature(l_p, l_q, l_r, l_s, k))) < 1e-10


# --- 3. The real-harmonic convention is the integral library's -------------------------------

def test_the_real_harmonics_are_the_ones_the_integral_library_uses():
    """⚠ **The only check of the sign convention, and it needs the library to be present.**

    A basis function is a radial factor times a real solid harmonic, so on a sphere of fixed
    radius every function of one contracted shell must be *the same constant* times this
    module's real harmonic for its own ``m``. A convention mismatch in one ``m`` channel shows
    up as one column with the wrong sign — invisible in any norm, and fatal to every parameter
    with an odd number of indices in that channel.

    The library's within-shell ordering is handled here rather than assumed: a ``p`` shell is
    stored ``px, py, pz``, which is ``m = +1, -1, 0``.
    """
    from pyscf import gto

    mol = gto.M(atom="Ce 0 0 0", basis="x2c-SVPall-2c", verbose=0)
    rng = np.random.default_rng(0)
    directions = rng.normal(size=(16, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    values = mol.eval_gto("GTOval_sph", 1.3 * directions)
    cos_theta, phi = directions[:, 2], np.arctan2(directions[:, 1], directions[:, 0])

    checked = set()
    offset = 0
    for shell in range(mol.nbas):
        l = mol.bas_angular(shell)
        harmonics = _real_harmonics(l, cos_theta, phi).real.T          # (npt, 2l+1) ascending m
        order = [1, -1, 0] if l == 1 else list(range(-l, l + 1))       # the library's own
        reference = harmonics[:, [m + l for m in order]]
        for _ in range(mol.bas_nctr(shell)):
            block = values[:, offset:offset + 2 * l + 1]
            offset += 2 * l + 1
            mask = np.abs(reference) > 1e-3
            ratio = block[mask] / reference[mask]
            if abs(ratio.mean()) < 1e-6:
                continue                       # a diffuse contraction, numerically empty here
            assert np.std(ratio) < 1e-10 * abs(ratio.mean()), (
                "the {} functions of shell {} are not one constant times the module's real "
                "harmonics".format("spdfghi"[l], shell))
            checked.add(l)
    assert checked >= {0, 1, 2, 3, 4}, "the check must reach the f and g shells, not only s"


# --- 4. Identities the tensors must satisfy --------------------------------------------------

def test_the_tensors_carry_the_permutational_symmetry_of_the_integral():
    """Over real orbitals ``(pq|rs) = (qp|rs) = (pq|sr) = (rs|pq)``, and none of those
    permutations changes which radial parameter multiplies it — so each is a symmetry of
    ``A^k`` itself, and a violated one means the index placement is wrong."""
    for (l_p, l_q, l_r, l_s) in [(3, 3, 3, 3), (3, 2, 2, 3), (2, 1, 1, 2), (3, 3, 0, 2)]:
        for k in admissible_k(l_p, l_q, l_r, l_s):
            a = angular_tensor(l_p, l_q, l_r, l_s, k)
            bra = angular_tensor(l_q, l_p, l_r, l_s, k)
            ket = angular_tensor(l_p, l_q, l_s, l_r, k)
            both = angular_tensor(l_r, l_s, l_p, l_q, k)
            assert np.allclose(a, np.transpose(bra, (1, 0, 2, 3)))
            assert np.allclose(a, np.transpose(ket, (0, 1, 3, 2)))
            assert np.allclose(a, np.transpose(both, (2, 3, 0, 1)))


def test_a_filled_channel_contributes_only_through_the_monopole():
    """Tracing the electron-1 pair over a whole shell leaves ``(2l+1)`` at ``k = 0`` and
    exactly zero above it, times the identity on the electron-2 pair.

    This is the statement that a closed shell has no multipole moment, it is what makes a
    configuration-average energy depend on ``F^0`` alone between closed shells, and it is a
    global property of the tensor rather than of any one element."""
    for (l_p, l_r, l_s) in [(3, 2, 2), (1, 3, 3), (2, 0, 0), (3, 3, 1)]:
        for k in admissible_k(l_p, l_p, l_r, l_s):
            traced = np.einsum("ppcd->cd", angular_tensor(l_p, l_p, l_r, l_s, k))
            expected = ((2 * l_p + 1) * np.eye(2 * l_r + 1)
                        if k == 0 and l_r == l_s else np.zeros((2 * l_r + 1, 2 * l_s + 1)))
            assert np.allclose(traced, expected, atol=1e-12)


def test_the_complex_to_real_transform_is_unitary_and_gives_x_y_z():
    """Unitarity is what makes the rotation of the tensors norm-preserving; the ``l = 1`` rows
    are the anchor of the sign convention, ``(x, y, z)`` with positive coefficients."""
    for l in range(5):
        u = complex_to_real(l)
        assert np.allclose(u @ u.conj().T, np.eye(2 * l + 1))
    root_half = 1 / math.sqrt(2)
    u = complex_to_real(1)                                       # rows are m = -1, 0, +1
    assert np.allclose(u[2], [root_half, 0.0, -root_half])       # x  ~ (Y_-1 - Y_+1)/sqrt2
    assert np.allclose(u[0], [1j * root_half, 0.0, 1j * root_half])   # y
    assert np.allclose(u[1], [0.0, 1.0, 0.0])                    # z  = Y_0
