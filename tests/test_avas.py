"""Tier 0/1: the AVAS active-space construction (``kuiva.mcscf.avas``).

What can go wrong here, and what each test is chosen to fail on
--------------------------------------------------------------
AVAS *rotates* orbitals, which is the difference between it and every other selection route
in this program — and a wrong rotation produces an orthonormal set of the right shape that
starts a calculation and converges. Three properties are therefore asserted directly, each
of which a plausible-looking wrong implementation breaks:

* **the density does not move.** The rotation is confined to groups of equal occupation, so
  the reference density — and with it the SCF energy everything downstream is built on — is
  invariant to machine precision. Rotating "the occupied space" as one block instead would
  pass every other check here and silently change the reference of an open-shell system.
* **Kramers pairing survives.** The barred partners transform with the *conjugate* rotation
  because time reversal is antiunitary; using the same rotation for both is the mistake that
  is invisible in the norm, in the orthonormality and in the density
  (:func:`test_rotating_the_barred_partners_with_v_instead_of_conj_v_breaks_pairing` shows
  it breaking, so the test can fail).
* **the eigenvalues mean what they say.** The projector's trace is the number of reference
  orbitals, so the eigenvalues sum to it exactly — an independent check on the whole
  construction that no threshold or ordering choice can satisfy by accident.

Tolerances: 1e-12 throughout on the invariances, which are exact linear algebra (they come
out at 1e-16); the eigenvalue sum rule at 1e-10.
"""
import importlib

import numpy as np
import pytest

import kuiva
from kuiva.interface.api import avas_active_space
from kuiva.spinor.expand import (fold_to_kramers_pairs, rotate_kramers_pairs,
                                 spin_block_diagonal, time_reverse)

avas_mod = importlib.import_module("kuiva.mcscf.avas")
avas = avas_mod.avas

EXACT = 1e-12
BASIS = "x2c-SVPall-2c"


# --- the spinor-convention helpers the rotation is built on ---------------------------------

def test_the_pair_fold_is_exact_for_a_spin_free_operator():
    """A spin-free operator lifted to two components folds back with zero residual."""
    rng = np.random.default_rng(5)
    a = rng.normal(size=(4, 4))
    a = 0.5 * (a + a.T)
    # In the interleaved convention 1_2 (x) A is the interleaved Kronecker product.
    lifted = np.zeros((8, 8), complex)
    for p in range(4):
        for q in range(4):
            lifted[2 * p, 2 * q] = lifted[2 * p + 1, 2 * q + 1] = a[p, q]
    folded, residual = fold_to_kramers_pairs(lifted)
    assert residual < EXACT
    assert np.abs(folded - a).max() < EXACT


def test_the_pair_fold_reports_rather_than_hides_a_spin_dependent_operator():
    """⚠ The residual is returned, not checked here: a folded operator that should not have
    been is Hermitian, plausible and wrong, so every caller has to make the decision."""
    a = np.zeros((4, 4), complex)
    a[0, 1] = a[1, 0] = 1.0                     # couples a pair's two members
    _, residual = fold_to_kramers_pairs(a)
    assert residual == pytest.approx(1.0)


def test_rotating_the_barred_partners_with_v_instead_of_conj_v_breaks_pairing():
    """⚠ The mechanism test for the conjugation trap in the pair rotation.

    Both rotations give an orthonormal set; only the conjugate one keeps column ``2m+1`` the
    time reverse of column ``2m``. Asserting the wrong one *fails* is what makes the right
    one's success meaningful.
    """
    rng = np.random.default_rng(9)
    nbas, npair = 5, 3
    base = np.linalg.qr(rng.normal(size=(nbas, nbas)))[0][:, :npair]
    c = np.zeros((2 * nbas, 2 * npair), complex)
    for m in range(npair):
        c[:nbas, 2 * m] = base[:, m]
        c[:, 2 * m + 1] = time_reverse(c[:, 2 * m:2 * m + 1]).ravel()
    v = np.linalg.qr(rng.normal(size=(npair, npair)) + 1j * rng.normal(size=(npair, npair)))[0]

    def pairing_error(x):
        return float(np.abs(1.0 - np.abs(np.sum(np.conj(x[:, 1::2]) * time_reverse(x[:, ::2]),
                                                axis=0))).max())

    right = rotate_kramers_pairs(c, v, np.arange(2 * npair))
    wrong = np.array(c, copy=True)
    wrong[:, 0::2] = c[:, 0::2] @ v
    wrong[:, 1::2] = c[:, 1::2] @ v                       # the trap: v, not conj(v)
    assert pairing_error(right) < EXACT
    assert pairing_error(wrong) > 0.1
    # both are orthonormal, which is why the mistake survives every ordinary check
    for x in (right, wrong):
        assert np.abs(x.conj().T @ x - np.eye(2 * npair)).max() < EXACT


def test_the_pair_rotation_refuses_columns_that_are_not_whole_pairs():
    c = np.zeros((4, 4), complex)
    with pytest.raises(ValueError, match="whole Kramers pairs"):
        rotate_kramers_pairs(c, np.eye(1), [1, 2])


# --- AVAS on a real molecule ----------------------------------------------------------------

@pytest.fixture(scope="module")
def water():
    """H2O with the free-atom reference orbitals ingested.

    ``screening="none"``: nothing here is a statement about spin-orbit coupling, and the
    suite may not depend on a warm AMF cache. ``atomic_reference=True`` is what AVAS projects
    onto and is the one thing this fixture exists to switch on.
    """
    mol = kuiva.Molecule([("O", (0.0, 0.0, 0.117)), ("H", (0.0, 0.757, -0.469)),
                          ("H", (0.0, -0.757, -0.469))], basis=BASIS)
    scf = kuiva.ScalarSCF(mol, memory_gb=8.0, screening="none",
                          atomic_reference=True).run()
    return kuiva.Reference(scf).run()


def run_avas(reference, **kwargs):
    r = reference.reference
    return avas(r.spinors_in_ao(), r.data.s_ao, r.ao_layout, r.data.atomic_reference,
                r.data.nelec_total, occupation=r.spinors.occ, **kwargs)


def test_the_rotation_leaves_the_reference_density_exactly_where_it_was(water):
    """⚠ The property the occupation grouping exists for. A rotation that mixed occupations
    would change the SCF density everything downstream is built on, and nothing but this
    would notice."""
    r = water.reference
    result = run_avas(water, atom="O", l="p")
    c0, occ = r.spinors_in_ao(), r.spinors.occ
    d0 = (c0 * occ) @ c0.conj().T
    d1 = (result.coeff * occ) @ result.coeff.conj().T
    assert np.abs(d0 - d1).max() < EXACT


def test_the_rotated_set_stays_orthonormal_and_kramers_paired(water):
    r = water.reference
    result = run_avas(water, atom="O", l="p")
    s2 = spin_block_diagonal(np.asarray(r.data.s_ao))
    gram = result.coeff.conj().T @ s2 @ result.coeff
    assert np.abs(gram - np.eye(gram.shape[0])).max() < 1e-10
    overlap = np.sum(np.conj(result.coeff[:, 1::2]) * (s2 @ time_reverse(result.coeff[:, ::2])),
                     axis=0)
    assert np.abs(1.0 - np.abs(overlap)).max() < 1e-10
    assert result.fold_residual < EXACT


def test_the_eigenvalues_sum_to_the_number_of_reference_orbitals(water):
    """The projector's trace is the dimension of the span it projects onto — an independent
    check on the projector, the fold and the per-group diagonalization together."""
    result = run_avas(water, atom="O", l="p")
    assert float(result.eigenvalues.sum()) == pytest.approx(3.0, abs=1e-10)


def test_the_oxygen_p_space_is_the_three_lone_pair_and_bonding_combinations(water):
    """Water's O 2p character sits almost entirely in the occupied space, so AVAS takes three
    occupied pairs and the eigenvalue gap at the cut is large."""
    result = run_avas(water, atom="O", l="p")
    assert result.n_pairs == 3
    assert np.allclose(result.occupations[result.selected], 2.0)
    assert result.gap > 0.5
    assert result.space.n_active == 6


def test_the_selection_is_one_contiguous_block_around_the_fermi_level(water):
    """⚠ What the per-group ordering exists for, and it is not cosmetic.

    An occupied group is ordered by ascending projection and an empty one by descending, so
    the orbitals carrying the character meet at the occupied/virtual boundary. Ordering every
    group the same way instead puts the most character-rich *occupied* orbital at column 0,
    below the core: a valid active space on paper, with an orbital layout nothing downstream
    expects. The threshold here is low on purpose, so that the selection spans both groups
    and the boundary is actually tested.
    """
    result = run_avas(water, atom="O", l="p", threshold=0.05)
    selected = np.sort(np.asarray(result.selected))
    assert selected[-1] - selected[0] == selected.size - 1
    occ = result.occupations[selected]
    assert occ[0] >= occ[-1]                          # occupied first, then virtual
    assert np.any(occ > 1.5) and np.any(occ < 0.5)    # the block really does straddle


def test_a_threshold_nothing_clears_says_what_the_projections_were(water):
    with pytest.raises(ValueError, match="largest projections"):
        run_avas(water, atom="O", l="p", threshold=1.5)


def test_max_pairs_refuses_rather_than_returning_a_too_large_space(water):
    """⚠ An AVAS whose threshold is slightly too low returns a plausible active space one or
    two pairs too big, and the cost of that is discovered when the CI runs."""
    with pytest.raises(ValueError, match="max_pairs"):
        run_avas(water, atom="O", l="p", threshold=0.05, max_pairs=3)


def test_a_shell_the_reference_basis_cannot_supply_is_refused(water):
    """Asking for more shells of an ``l`` than the free atom has is a refusal, not a guess."""
    with pytest.raises(ValueError, match="shell"):
        run_avas(water, atom="O", l="p", n_shells=99)


def test_a_set_that_is_not_kramers_paired_is_refused_and_names_the_likely_cause(water):
    """⚠ An unrestricted reference gives spinors that are orthonormal but **not** Kramers
    paired, and the fold onto pairs is meaningless there. It must refuse rather than average
    two unrelated columns into one pair — the result would be Hermitian and plausible."""
    r = water.reference
    c = np.array(r.spinors_in_ao(), copy=True)
    c[:, 1] = c[:, 3]                                  # break the pairing of one pair only
    with pytest.raises(ValueError, match="Kramers-pair structure"):
        avas(c, r.data.s_ao, r.ao_layout, r.data.atomic_reference, r.data.nelec_total,
             atom="O", l="p", occupation=r.spinors.occ)


def test_a_missing_atomic_reference_names_the_knob_that_supplies_it(water):
    r = water.reference
    with pytest.raises(ValueError, match="atomic_reference=True"):
        avas(r.spinors_in_ao(), r.data.s_ao, r.ao_layout, None, r.data.nelec_total,
             atom="O", l="p", occupation=r.spinors.occ)


def test_the_double_shell_request_projects_onto_two_shells(water):
    """``n_shells=2`` is how the correlating shell is named — the case a character threshold
    cannot find, because the second shell is diffuse and covalent."""
    one = run_avas(water, atom="O", l="p", n_shells=1)
    two = run_avas(water, atom="O", l="p", n_shells=2)
    assert float(two.eigenvalues.sum()) == pytest.approx(6.0, abs=1e-10)
    assert two.n_pairs > one.n_pairs
    assert "2 shell(s)" in two.reference


# --- the stage surface ----------------------------------------------------------------------

def test_the_stage_refuses_avas_together_with_another_selection(water):
    with pytest.raises(ValueError, match="exactly one"):
        kuiva.CASSCF(water, character=("O", "p"), n_active=6, avas=dict(atom="O", l="p"))


def test_the_stage_refuses_avas_on_a_cheap_ci_upstream(water):
    """⚠ The cheap CI's natural occupations are all distinct, so every group would hold one
    pair and the rotation would be the identity — an AVAS that silently did nothing."""
    cheap = kuiva.CheapCI(water, character=("O", "p"), n_active=6)
    cheap._ran = True                              # no need to pay for the pre-optimization
    with pytest.raises(ValueError, match="CheapCI"):
        kuiva.CASSCF(cheap, avas=dict(atom="O", l="p"))


def test_the_stage_refuses_n_active_beside_avas(water):
    """AVAS chooses the size from the projection spectrum; a second statement of it would be
    two answers to one question."""
    with pytest.raises(ValueError, match="max_pairs"):
        kuiva.CASSCF(water, avas=dict(atom="O", l="p"), n_active=6)


def test_the_stage_refuses_an_unknown_avas_option(water):
    with pytest.raises(TypeError, match="avas"):
        kuiva.CASSCF(water, avas=dict(atom="O", l="p", nonsense=1))


def test_the_api_route_defaults_to_the_references_own_orbitals(water):
    """``api.avas_active_space`` is the same construction with the reference's orbitals and
    occupations filled in, and must agree with the explicit call element for element."""
    direct = run_avas(water, atom="O", l="p")
    through = avas_active_space(water.reference, atom="O", l="p", report=False)
    assert np.abs(direct.coeff - through.coeff).max() < EXACT
    assert list(direct.space.spaces.active) == list(through.space.spaces.active)


def test_an_avas_space_carries_no_symmetry_labels(water):
    """⚠ The labels belong to the guess spinors and AVAS has rotated them; carrying them
    across would attach a label to an orbital it no longer describes."""
    result = avas_active_space(water.reference, atom="O", l="p", report=False)
    assert getattr(result.space, "labels", None) is None
