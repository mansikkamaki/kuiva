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


# --- the count-stated mode ------------------------------------------------------------------

def test_the_count_mode_reproduces_the_threshold_modes_selection(water):
    """⚠ The two rules must agree wherever they are both right, or they are two definitions
    of the shell rather than two ways of stating one.

    Water's O 2p projections are 0.995, 0.941, 0.878 and then 0.122, so "everything above
    0.2" and "the three of largest projection" are the same three pairs -- and the rotated
    orbitals must come out bitwise identical, because the rotation happens before either
    rule is applied.
    """
    by_threshold = run_avas(water, atom="O", l="p", threshold=0.2)
    by_count = run_avas(water, atom="O", l="p", n_pairs=3)
    assert list(by_count.selected) == list(by_threshold.selected)
    assert by_count.gap == by_threshold.gap
    assert np.array_equal(by_count.coeff, by_threshold.coeff)
    assert list(by_count.space.spaces.active) == list(by_threshold.space.spaces.active)


def test_the_count_mode_records_which_rule_chose_the_space(water):
    """⚠ A stored product has to say how its active space was chosen: "seven pairs because
    that is the shell" and "seven pairs because they cleared 0.2" are different statements
    and only one of them is reproducible in another basis."""
    by_count = run_avas(water, atom="O", l="p", n_pairs=3)
    assert (by_count.mode, by_count.cut) == ("count", 3.0)
    assert "count stated" in by_count.space.description
    by_threshold = run_avas(water, atom="O", l="p", threshold=0.2)
    assert (by_threshold.mode, by_threshold.cut) == ("threshold", 0.2)
    assert "threshold" in by_threshold.space.description


def test_a_count_and_a_threshold_together_are_refused(water):
    """Two statements of what the active space *is*; a run that silently preferred one would
    not be reproducible from its description."""
    with pytest.raises(ValueError, match="not both"):
        run_avas(water, atom="O", l="p", threshold=0.2, n_pairs=3)
    with pytest.raises(ValueError, match="max_pairs"):
        run_avas(water, atom="O", l="p", n_pairs=3, max_pairs=5)


def test_a_count_larger_than_the_orbital_set_is_refused(water):
    with pytest.raises(ValueError, match="more Kramers pairs than"):
        run_avas(water, atom="O", l="p", n_pairs=999)
    with pytest.raises(ValueError, match="must be positive"):
        run_avas(water, atom="O", l="p", n_pairs=0)


def test_the_projection_without_a_space_is_the_same_projection(water):
    """``avas_projection`` exists for the caller that must own the electron count -- a shell
    empty in the reference has no aufbau one. It may differ from ``avas`` in nothing else."""
    from kuiva.mcscf.avas import avas_projection

    r = water.reference
    projection = avas_projection(r.spinors_in_ao(), r.data.s_ao, r.ao_layout,
                                 r.data.atomic_reference, atom="O", l="p",
                                 occupation=r.spinors.occ, n_pairs=3)
    full = run_avas(water, atom="O", l="p", n_pairs=3)
    assert projection.space is None and full.space is not None
    assert np.array_equal(projection.coeff, full.coeff)
    assert np.array_equal(projection.eigenvalues, full.eigenvalues)
    assert list(projection.selected) == list(full.selected)
    assert projection.reference_statement == full.space.description
    projection.report()                     # must not need a space to report itself


# --- several shells: one projection onto the union ----------------------------------------------

def _span(result, coeff=None, pairs=None):
    coeff = result.coeff if coeff is None else coeff
    pairs = result.selected if pairs is None else pairs
    return coeff[:, np.concatenate([[2 * int(p), 2 * int(p) + 1] for p in pairs])]


def _principal(reference, a, b):
    s2 = spin_block_diagonal(np.asarray(reference.reference.data.s_ao))
    return np.linalg.svd(a.conj().T @ s2 @ b, compute_uv=False)


def test_one_listed_shell_is_bitwise_the_single_shell_form(water):
    """``shells=[(atom, l)]`` may not be a second construction of the one-shell projector:
    every committed single-shell space goes through the same code, so they must agree to
    the bit."""
    single = run_avas(water, atom="O", l="p", n_pairs=3)
    listed = run_avas(water, shells=[("O", "p")], n_pairs=3)
    assert np.array_equal(single.coeff, listed.coeff)
    assert np.array_equal(single.eigenvalues, listed.eigenvalues)
    assert listed.component_projections is None and listed.components == ()


def test_the_union_is_one_projector_whatever_order_its_shells_are_listed_in(water):
    """The union's trace is the number of reference orbitals of all its shells, and its
    selection is a span of eigenvectors of one projector -- so listing the shells in the
    other order may change the labels and nothing else."""
    ab = run_avas(water, shells=[("O", "s"), ("O", "p")], n_pairs=4)
    ba = run_avas(water, shells=[("O", "p"), ("O", "s")], n_pairs=4)
    assert float(ab.eigenvalues.sum()) == pytest.approx(4.0, abs=1e-10)
    assert ab.gap > 0.5
    overlaps = _principal(water, _span(ab), _span(ba))
    assert np.abs(overlaps - 1.0).max() < 1e-10
    assert ab.space.n_active == 8


def test_two_projections_in_sequence_lose_the_first_selection(water):
    """⚠ **The mechanism the union exists for, and the test that it can fail.** A second
    AVAS call rotates inside the occupation groups, where the first call's selected pairs lie
    outside its projector's span -- degenerate at zero there, so they come back as whatever
    basis the diagonalization returns. The first selection is then not held by the columns
    the two calls name (on this system a column is even claimed twice), by an amount that
    differed between two identical runs."""
    from kuiva.mcscf.avas import avas_projection

    r = water.reference
    first = avas_projection(r.spinors_in_ao(), r.data.s_ao, r.ao_layout,
                            r.data.atomic_reference, atom="O", l="p",
                            occupation=r.spinors.occ, n_pairs=3)
    second = avas_projection(first.coeff, r.data.s_ao, r.ao_layout, r.data.atomic_reference,
                             atom=[1, 2], l="s", occupation=r.spinors.occ, n_pairs=2)
    named = sorted(set(first.selected.tolist()) | set(second.selected.tolist()))
    kept = _principal(water, _span(first), _span(first, second.coeff, named))
    assert float(kept.min()) < 0.99, kept
    union = run_avas(water, shells=[("O", "p"), ([1, 2], "s")], n_pairs=5)
    assert union.n_pairs == 5 and len(set(union.selected.tolist())) == 5


def test_a_union_degenerate_across_its_shells_is_separated_shell_by_shell():
    """⚠ The mechanism behind the far Ti(3+)/Ce(3+) pair. Where two shells do not overlap the
    union projects at eigenvalue 1 on both, the diagonalization returns ANY basis of that block
    (it returned pairs 96/4, 14/86 and 85/15 per cent Ti/Ce there), and an attribution by
    argmax is then the diagonalization's choice. The second diagonalization inside the block
    must hand back pure shell vectors, and must leave a non-degenerate eigenvector and the
    null space exactly where they were."""
    rng = np.random.default_rng(3)
    n = 7
    basis = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))[0]
    p_a = basis[:, :2] @ basis[:, :2].conj().T                      # shell A: 2 pairs
    p_b = basis[:, 2:5] @ basis[:, 2:5].conj().T                    # shell B: 3 pairs
    union = p_a + p_b
    w, v = np.linalg.eigh(union)
    order = np.argsort(-w)
    w, v = w[order], v[:, order]
    mixed = v[:, :5] @ np.linalg.qr(rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5)))[0]
    v = np.concatenate([mixed, v[:, 5:]], axis=1)                   # an arbitrary block basis
    out = avas_mod._separate_degenerate_shells(w, v, 1.0 * p_a + 2.0 * p_b)
    on_a = np.real(np.einsum("ip,ij,jp->p", out.conj(), p_a, out))
    on_b = np.real(np.einsum("ip,ij,jp->p", out.conj(), p_b, out))
    assert np.allclose(on_b[:3], 1.0, atol=1e-10) and np.allclose(on_a[3:5], 1.0, atol=1e-10)
    assert np.allclose(out.conj().T @ out, np.eye(n), atol=1e-12)
    assert np.allclose(out.conj().T @ union @ out, np.diag(w), atol=1e-10)
    assert np.array_equal(out[:, 5:], v[:, 5:]), "the null space is not the block's to move"
    before = np.real(np.einsum("ip,ij,jp->p", v.conj(), p_a, v))[:5]
    assert np.max(np.minimum(before, 1.0 - before)) > 1e-3, "the test's block must start mixed"


def test_each_pair_of_a_union_is_attributed_to_the_shell_it_projects_onto_most(water):
    """A diagonal read in the rotated orbitals, reported per shell in the statement -- never
    a second rotation, which would be the sequential defect again."""
    result = run_avas(water, shells=[("O", "s"), ("O", "p")], n_pairs=4)
    assert result.components and len(result.components) == 2
    assert result.component_projections.shape == (2, result.eigenvalues.size)
    owners = result.owners()
    assert sorted(np.bincount(owners, minlength=2).tolist()) == [1, 3]
    assert int(np.count_nonzero(owners == 0)) == 1, "one O 2s pair, three O 2p pairs"
    assert "union of" in result.space.description
    assert "per shell 1 + 3" in result.space.description
    result.report()                            # the per-shell column must print


def test_a_single_shell_and_a_union_together_are_refused(water):
    with pytest.raises(ValueError, match="not both"):
        run_avas(water, atom="O", l="p", shells=[("O", "s")], n_pairs=3)
    with pytest.raises(ValueError, match="more than one entry"):
        run_avas(water, shells=[("O", "p"), (0, "p")], n_pairs=3)
    with pytest.raises(ValueError, match="atom= and l="):
        run_avas(water, n_pairs=3)
    single = run_avas(water, atom="O", l="p", n_pairs=3)
    with pytest.raises(ValueError, match="no components"):
        single.owners()


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
