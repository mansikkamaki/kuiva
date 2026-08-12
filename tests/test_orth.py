"""Tier-0 tests for the orthonormal working basis.

The working basis is the object everything downstream is built on, so these are invariance
and consistency tests, not accuracy tests: X^T S X = 1 exactly, the generalized eigenproblem
in the AO basis and the ordinary one in the working basis must agree, and a deliberately
linearly dependent basis must lose exactly the dimensions it should.
"""
import numpy as np
import pytest

from kuiva.interface import Molecule
from kuiva.interface.pyscf_bridge import run_scalar_x2c
from kuiva.orth.canonical import (DEFAULT_THRESHOLD, canonical_orthogonalization,
                                  cholesky_orthogonalization, orthogonalize,
                                  project_orbitals, symmetric_orthogonalization)

ORTHO_TOL = 1e-12          # X^T S X = 1 is exact up to accumulation of rounding
EIG_TOL = 1e-9             # Eh; eigenvalues of the same operator in two bases


def random_overlap(n, seed=0, n_dependent=0):
    """A synthetic overlap matrix: SPD, unit diagonal, with ``n_dependent`` exact
    linear dependencies built in (a column repeated), which is what a near-coincident
    diffuse function pair does to a real basis."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n - n_dependent))
    if n_dependent:
        a = np.concatenate([a, a[:, :n_dependent]], axis=1)
    s = a.T @ a
    d = np.sqrt(np.diag(s))
    return s / np.outer(d, d)


# --- Tier 0: the defining property ------------------------------------------------------
@pytest.mark.parametrize("scheme", ["canonical", "symmetric", "cholesky"])
def test_orthonormality(scheme):
    s = random_overlap(20, seed=1)
    ob = orthogonalize(s, scheme)
    ident = ob.x.T @ s @ ob.x
    assert np.max(np.abs(ident - np.eye(ob.nwork))) < ORTHO_TOL


def test_x_dag_is_the_metric_adjoint():
    """``x_dag`` must map AO coefficients into the working basis, i.e. be X^T S, not X^T."""
    s = random_overlap(12, seed=2)
    ob = canonical_orthogonalization(s)
    assert np.max(np.abs(ob.x_dag - ob.x.T @ s)) < 1e-14
    # Round trip: a vector that lives in the working space comes back unchanged.
    v = np.random.default_rng(3).standard_normal((ob.nwork, 4))
    assert np.max(np.abs(ob.to_working(ob.to_ao(v)) - v)) < 1e-10


def test_linear_dependence_is_removed_and_reported(kuiva_caplog):
    n, dep = 16, 3
    s = random_overlap(n, seed=4, n_dependent=dep)
    caplog = kuiva_caplog
    with caplog.at_level("WARNING"):
        ob = canonical_orthogonalization(s)
    assert ob.n_dropped == dep                    # exactly the dependencies, no more
    assert ob.nwork == n - dep
    assert ob.largest_dropped <= DEFAULT_THRESHOLD
    assert ob.smallest_kept > DEFAULT_THRESHOLD
    assert any("linearly-dependent" in r.message for r in caplog.records)
    # Still an orthonormal basis of the retained space.
    assert np.max(np.abs(ob.x.T @ s @ ob.x - np.eye(ob.nwork))) < ORTHO_TOL


def test_symmetric_refuses_a_dependent_basis():
    """A rank-deficient square X would be a singular matrix wearing the name 'orthonormal'."""
    s = random_overlap(10, seed=5, n_dependent=2)
    with pytest.raises(ValueError, match="linearly dependent"):
        symmetric_orthogonalization(s)


def test_eigenvalues_are_basis_independent():
    """The physical content must not depend on the orthogonalization scheme (Tier 0)."""
    s = random_overlap(14, seed=6)
    rng = np.random.default_rng(7)
    h = rng.standard_normal((14, 14))
    h = 0.5 * (h + h.T)
    from scipy.linalg import eigh as geigh
    ref = geigh(h, s, eigvals_only=True)          # the generalized problem in the AO basis
    for scheme in ("canonical", "symmetric", "cholesky"):
        ob = orthogonalize(s, scheme)
        got = np.linalg.eigvalsh(ob.transform_operator(h))
        assert np.max(np.abs(np.sort(got) - np.sort(ref))) < EIG_TOL


def test_column_phase_is_deterministic():
    """Fixed phases keep checkpoint restarts and run-to-run comparisons reproducible."""
    s = random_overlap(11, seed=8)
    a = canonical_orthogonalization(s)
    b = canonical_orthogonalization(s.copy())
    assert np.array_equal(a.x, b.x)
    # and the convention itself: largest-magnitude element of each column of U is positive
    u = a.x * np.sqrt(a.s_eigenvalues[:a.nwork])
    idx = np.argmax(np.abs(u), axis=0)
    assert np.all(u[idx, np.arange(a.nwork)] > 0)


def test_descending_eigenvalue_order():
    s = random_overlap(13, seed=9, n_dependent=2)
    ob = canonical_orthogonalization(s)
    assert np.all(np.diff(ob.s_eigenvalues) <= 1e-14)     # descending: dropped ones are a tail


# --- Tier 1: against a real overlap matrix ---------------------------------------------
@pytest.fixture(scope="module")
def ne_data():
    mol = Molecule([("Ne", (0.0, 0.0, 0.0))], basis="x2c-SVPall-2c")
    return run_scalar_x2c(mol, fitting="conventional", screening="none")


def test_real_basis_is_well_conditioned(ne_data):
    ob = canonical_orthogonalization(ne_data.s_ao)
    assert ob.n_dropped == 0                       # a small segmented set has no dependence
    assert ob.condition_number < 1e6


def test_scf_orbitals_project_without_loss(ne_data):
    """With nothing dropped, the projection into the working basis is exact and reversible."""
    ob = canonical_orthogonalization(ne_data.s_ao)
    c_work = project_orbitals(ob, ne_data.mo_coeff, ne_data.s_ao)
    assert np.max(np.abs(ob.to_ao(c_work) - ne_data.mo_coeff)) < 1e-10
    # MOs are orthonormal in the working basis without a metric.
    assert np.max(np.abs(c_work.T @ c_work - np.eye(c_work.shape[1]))) < 1e-10


def test_hcore_eigenvalues_survive_the_working_basis(ne_data):
    from scipy.linalg import eigh as geigh
    ref = geigh(ne_data.h_x2c, ne_data.s_ao, eigvals_only=True)
    ob = canonical_orthogonalization(ne_data.s_ao)
    got = np.linalg.eigvalsh(ob.transform_operator(ne_data.h_x2c))
    assert np.max(np.abs(np.sort(got) - np.sort(ref))) < 1e-8


def test_cholesky_orthogonalization_matches_canonical_spectrum(ne_data):
    a = orthogonalize(ne_data.s_ao, "canonical")
    b = orthogonalize(ne_data.s_ao, "cholesky")
    ea = np.linalg.eigvalsh(a.transform_operator(ne_data.h_x2c))
    eb = np.linalg.eigvalsh(b.transform_operator(ne_data.h_x2c))
    assert np.max(np.abs(ea - eb)) < 1e-9


def test_unknown_scheme_is_rejected():
    with pytest.raises(ValueError, match="unknown orthogonalization scheme"):
        orthogonalize(np.eye(3), "loewdin-ish")


# --- Tier 0: the cut drops whole degenerate groups --------------
#
# The mechanism, not only the observable: the observable is "the working basis has
# the dimension it should", the mechanism is "the retained space is invariant under whatever
# symmetry made those overlap eigenvalues degenerate". A partial group passes every dimension,
# trace and orthonormality check there is and still gives an anisotropic basis.

def overlap_with_a_degenerate_group(n_group, value, seed=7, n_other=6):
    """An overlap whose spectrum has an exactly ``n_group``-fold eigenvalue at ``value``.

    Built by conjugating a chosen spectrum with a random orthogonal matrix, which is what a
    symmetry-degenerate shell does to a real overlap without needing a real molecule.
    """
    rng = np.random.default_rng(seed)
    spectrum = np.concatenate([np.geomspace(1.0, 1e-2, n_other),
                               np.full(n_group, float(value))])
    q, _ = np.linalg.qr(rng.standard_normal((spectrum.size, spectrum.size)))
    return (q * spectrum) @ q.T


def test_a_degenerate_group_straddling_the_cut_is_dropped_whole(kuiva_caplog):
    # A 3-fold group sitting *on* the threshold, so a bare `evals > threshold` keeps part of
    # it: two members land a hair above 1e-7 and one a hair below.
    s = overlap_with_a_degenerate_group(3, 1e-7)
    evals = np.linalg.eigvalsh(s)[::-1]
    group = evals[np.abs(evals - 1e-7) / 1e-7 < 1e-6]
    assert group.size == 3                              # the planted group survived the build
    threshold = float(np.median(group))                 # cuts the group in half by rounding
    naive = int(np.count_nonzero(evals > threshold))
    assert 0 < naive - (evals.size - 3) < 3, "the planted group must straddle for this test"

    with kuiva_caplog.at_level("WARNING"):
        ob = canonical_orthogonalization(s, threshold)
    assert ob.nwork == evals.size - 3                   # the whole group went, not part of it
    assert any("degenerate group" in r.message for r in kuiva_caplog.records)
    assert np.max(np.abs(ob.x.T @ s @ ob.x - np.eye(ob.nwork))) < ORTHO_TOL


def test_a_clean_cut_is_untouched_by_the_group_rule():
    """The rule is a guard: where no group straddles, the count is the bare threshold's.

    This is what makes the change reference-neutral — measured over every system and basis in
    the project, no cut in 1e-5..1e-8 straddles a group (measured).
    """
    rng = np.random.default_rng(11)
    for seed in range(50):
        s = random_overlap(20, seed=seed, n_dependent=rng.integers(0, 4))
        evals = np.linalg.eigvalsh(s)[::-1]
        ob = canonical_orthogonalization(s)
        assert ob.nwork == int(np.count_nonzero(evals > DEFAULT_THRESHOLD))


def test_the_metric_projection_uses_the_same_rule_as_the_orthogonalization():
    """The X2C metric cut and this one must be the same operation (one shared implementation)."""
    from kuiva.x2c.decouple import metric_keep_mask

    s = overlap_with_a_degenerate_group(3, 1e-7, seed=3)
    ascending = np.linalg.eigvalsh(s)
    threshold = float(np.median(ascending[np.abs(ascending - 1e-7) / 1e-7 < 1e-6]))
    mask = metric_keep_mask(ascending, threshold)
    assert int(mask.sum()) == ascending.size - 3
    assert mask[-1] and not mask[0]                     # ascending: the largest are kept
    # and the two entry points agree on the count
    assert int(mask.sum()) == canonical_orthogonalization(s, threshold).nwork
