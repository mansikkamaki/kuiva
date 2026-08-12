"""Tier-0 tests for the real decomposition of a spinor density.

Everything here is exact by construction, which is the reason the decomposition was chosen
over a fit: ``rho = sum_k w_k (u_k . chi)^2`` holds to machine precision or the implementation
is wrong. The three properties that matter are established separately, because a decomposition
can satisfy any two of them while being useless:

1. it reproduces the density matrix it came from (the algebra);
2. it reproduces the density *as a function of position*, checked on a real Gaussian basis on
   a real grid (the physics — an AO-ordering or normalization error survives (1) and dies
   here);
3. Kramers partners give the same density, so one picture per pair is complete.

The counter-test is the point of the module: the "obvious" per-coefficient square root is
measured to be wrong by ~50% of the peak density, so the exactness above is not a technicality.
"""
import numpy as np
import pytest

from kuiva.spinor.density import (decompose_density, decompose_spinor_density,
                                  kramers_pair_density, spinor_density_matrix)
from kuiva.spinor.expand import time_reverse

#: Machine precision is the right tolerance: this is an eigendecomposition of a rank-<=4
#: matrix, not an approximation of anything.
EXACT = 1e-12


@pytest.fixture(scope="module")
def metric():
    """A random positive-definite overlap and a normalized spinor in it."""
    rng = np.random.default_rng(11)
    nao = 8
    a = rng.standard_normal((nao, nao))
    s = a @ a.T + nao * np.eye(nao)
    c = rng.standard_normal((2 * nao, 1)) + 1j * rng.standard_normal((2 * nao, 1))
    norm = (c[:nao].conj().T @ s @ c[:nao] + c[nao:].conj().T @ s @ c[nao:]).real[0, 0]
    return s, c / np.sqrt(norm)


def test_components_reproduce_the_density_matrix(metric):
    s, c = metric
    dec = decompose_spinor_density(c, s)
    assert np.max(np.abs(dec.density_matrix() - spinor_density_matrix(c))) < EXACT


def test_weights_sum_to_the_norm(metric):
    s, c = metric
    dec = decompose_spinor_density(c, s)
    assert abs(dec.weights.sum() - 1.0) < EXACT
    assert np.all(dec.weights > 0.0)                 # a density is positive semidefinite


def test_components_are_s_orthonormal(metric):
    s, c = metric
    dec = decompose_spinor_density(c, s)
    gram = dec.components.T @ s @ dec.components
    assert np.max(np.abs(gram - np.eye(dec.n_components))) < EXACT


def test_rank_never_exceeds_four_per_spinor(metric):
    s, c = metric
    assert decompose_spinor_density(c, s).n_components <= 4


def test_a_real_spinor_is_a_single_component(metric):
    """The SOC-free guess: one real orbital, so one picture is the whole density."""
    s, c = metric
    nao = s.shape[0]
    real = np.zeros((2 * nao, 1))
    real[:nao, 0] = np.real(c[:nao, 0])
    dec = decompose_spinor_density(real, s)
    assert dec.n_components == 1
    assert dec.leading_weight == pytest.approx(dec.weights.sum(), abs=EXACT)


def test_kramers_partners_have_the_same_density(metric):
    """Why one entry per pair is complete rather than a simplification."""
    s, c = metric
    one = decompose_spinor_density(c, s)
    other = decompose_spinor_density(time_reverse(c), s)
    assert np.max(np.abs(one.density_matrix() - other.density_matrix())) < EXACT
    assert np.max(np.abs(np.sort(one.weights) - np.sort(other.weights))) < EXACT


def test_a_kramers_pair_stays_rank_four(metric):
    """The pair's (8, 8) Gram matrix has four zero eigenvalues, and they must be dropped."""
    s, c = metric
    pair = np.hstack([c, time_reverse(c)])
    dec = decompose_density(pair, [1.0, 1.0], s)
    assert dec.n_components <= 4
    assert dec.weights.sum() == pytest.approx(2.0, abs=EXACT)
    single = decompose_spinor_density(c, s)
    assert np.max(np.abs(dec.density_matrix() - 2.0 * single.density_matrix())) < EXACT


def test_kramers_pair_density_is_the_single_spinor_one(metric):
    s, c = metric
    assert np.max(np.abs(kramers_pair_density(c, s).density_matrix()
                         - decompose_spinor_density(c, s).density_matrix())) < EXACT


def test_occupation_weights_scale_the_density(metric):
    s, c = metric
    dec = decompose_density(c, [0.25], s)
    assert dec.weights.sum() == pytest.approx(0.25, abs=EXACT)


def test_orthonormal_scalar_basis_needs_no_overlap(metric):
    """``s_ao=None`` is the working basis, and must agree with passing the identity."""
    _, c = metric
    nao = c.shape[0] // 2
    a = decompose_spinor_density(c, None)
    b = decompose_spinor_density(c, np.eye(nao))
    assert np.max(np.abs(a.density_matrix() - b.density_matrix())) < EXACT


def test_zero_spinor_gives_no_components():
    dec = decompose_spinor_density(np.zeros(12, dtype=complex))
    assert dec.n_components == 0


def test_odd_row_count_rejected():
    with pytest.raises(ValueError, match="even number of rows"):
        decompose_density(np.zeros((7, 1), dtype=complex))


def test_negative_occupations_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        decompose_density(np.zeros((8, 1), dtype=complex), [-1.0])


# --- The physics: the density as a function of position ------------------------------------

@pytest.fixture(scope="module")
def gaussian_system():
    """A real Gaussian basis, its overlap, an AO evaluator, and an m_j-like d spinor.

    ``(d_{x2-y2} + i d_{xy})/sqrt(2)`` is ``|l=2, m=+2>``: the standard case where the density
    is a torus and *no* single real orbital in this basis squares to it.
    """
    from pyscf import gto

    mol = gto.M(atom=[["Ti", (0, 0, 0)], ["Cl", (0, 0, 2.3)]], basis="x2c-SVPall-2c",
                spin=1, verbose=0)
    s = mol.intor("int1e_ovlp")
    nao = mol.nao
    labels = [lbl.strip() for lbl in mol.ao_labels()]
    i_xy = labels.index([l for l in labels if l.endswith("3dxy")][0])
    i_x2 = labels.index([l for l in labels if l.endswith("3dx2-y2")][0])
    c = np.zeros((2 * nao, 1), dtype=complex)
    c[i_x2, 0] = 1.0 / np.sqrt(2.0)
    c[i_xy, 0] = 1j / np.sqrt(2.0)
    norm = (c[:nao].conj().T @ s @ c[:nao] + c[nao:].conj().T @ s @ c[nao:]).real[0, 0]
    return mol, s, c / np.sqrt(norm)


def test_mj_eigenfunction_needs_exactly_two_components(gaussian_system):
    """0.5/0.5 — the case that rules out writing one orbital per spinor."""
    _, s, c = gaussian_system
    dec = decompose_spinor_density(c, s)
    assert dec.n_components == 2
    assert np.allclose(dec.weights, [0.5, 0.5], atol=1e-10)


def test_density_is_exact_on_a_grid(gaussian_system):
    """The load-bearing test: the *function*, not the matrix.

    Tolerance is machine precision relative to the peak density. An AO-ordering error, a
    normalization error or a wrong metric all survive the algebraic tests and fail here.
    """
    mol, s, c = gaussian_system
    nao = mol.nao
    rng = np.random.default_rng(5)
    pts = rng.uniform(-3.0, 3.0, size=(400, 3))
    ao = mol.eval_gto("GTOval_sph", pts)

    rho = np.abs(ao @ c[:nao, 0]) ** 2 + np.abs(ao @ c[nao:, 0]) ** 2
    dec = decompose_spinor_density(c, s)
    rho_components = ((ao @ dec.components) ** 2 * dec.weights).sum(axis=1)
    assert np.max(np.abs(rho_components - rho)) < 1e-12 * max(rho.max(), 1.0)


def test_per_coefficient_square_root_is_wrong(gaussian_system):
    """The counter-test: ``sqrt(|c_a|^2 + |c_b|^2)`` per coefficient is not sqrt(rho).

    Recorded as an assertion rather than a comment because it is the tempting implementation,
    it produces a plausible picture, and nothing downstream would catch it. Measured error is
    ~50% of the peak density.
    """
    mol, s, c = gaussian_system
    nao = mol.nao
    rng = np.random.default_rng(6)
    pts = rng.uniform(-3.0, 3.0, size=(200, 3))
    ao = mol.eval_gto("GTOval_sph", pts)

    rho = np.abs(ao @ c[:nao, 0]) ** 2 + np.abs(ao @ c[nao:, 0]) ** 2
    naive = np.sqrt(np.abs(c[:nao, 0]) ** 2 + np.abs(c[nao:, 0]) ** 2)
    assert np.max(np.abs((ao @ naive) ** 2 - rho)) > 0.1 * rho.max()
