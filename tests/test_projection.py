"""Tier 0/1: carrying an orbital set from one basis set onto another (``kuiva.orth.project``).

The failure mode this feature has is the one worth designing tests around: **every way of
getting a basis projection wrong still produces an orthonormal orbital set of the right shape
that starts a calculation which converges**. It happened during development — one signed
comparison the wrong way round delivered a set whose "inactive" orbitals were high-lying
virtuals, and the only visible consequence was a CASSCF that took seven times as many
iterations to reach the same energy.

So the assertions here are about the properties the construction is supposed to *have*, each
chosen to be something a plausible-looking wrong implementation would fail:

* the delivered set is orthonormal **in the target's own metric**, not in some other one;
* projecting a set onto **its own basis** is the identity, to machine precision — the one
  case where the right answer is known independently of everything in the module;
* **Kramers pairing survives**, for all three schemes, because the projector commutes with
  time reversal and the orthonormalizations preserve self-duality — properties, not accidents;
* the completed columns really are the target's own low-lying orbitals, in **energy order**
  (this is what that bug destroyed while changing nothing else);
* the invariant diagnostics say what they claim: the principal overlaps are the cosines of the
  principal angles, and the retained norm is the norm the target basis holds;
* the refusals fire — a target too small to hold the active space, a mismatched molecule.

Tolerances: orthonormality 1e-12 (a projection is a few GEMMs and one eigendecomposition on
matrices of a few hundred, so 1e-14 is what it produces); the identity projection 1e-10;
CASSCF energies 1e-8 Eh, the suite's energy tolerance, because a projected guess must reach
the *same* answer — a guess that moved it would be a defect and not an acceleration.
"""
import numpy as np
import pytest

import kuiva
from kuiva.interface.api import (casscf, project_to_basis, projected_active_space,
                                 spinor_reference)
from kuiva.interface.pyscf_bridge import cross_overlap
from kuiva.interface.stages import CASSCF, CheapCI, Reference, ScalarSCF
from kuiva.orth.project import SCHEMES, plan_columns, project_spinors
from kuiva.spinor.expand import time_reverse

SMALL = "x2c-SVPall-2c"
LARGE = "x2c-TZVPall-2c"
E_TOL = 1e-8                        # suite energy tolerance [Eh]


def boron(basis: str) -> kuiva.Molecule:
    return kuiva.Molecule([("B", (0.0, 0.0, 0.0))], basis=basis, spin=1)


@pytest.fixture(scope="module")
def refs():
    """The same atom in two bases. ``screening="none"``: the two-electron picture change is
    pure cost here — nothing in this file is a statement about spin-orbit coupling, and the
    suite may not depend on a warm AMF cache."""
    out = {}
    for key, basis in (("small", SMALL), ("large", LARGE)):
        out[key] = spinor_reference(boron(basis), memory_gb=8.0, screening="none")
    return out


def metric(ref):
    """The two-component AO overlap: rows are spin-blocked, so it is ``1_2 (x) S``."""
    return np.kron(np.eye(2), np.asarray(ref.data.s_ao))


def orthonormality(ref, coeff):
    g = coeff.conj().T @ metric(ref) @ coeff
    return float(np.abs(g - np.eye(g.shape[0])).max())


# --- the column bookkeeping (pure integers, no integrals) -----------------------------------

def test_growing_appends_the_new_dimensions_to_the_virtual_space():
    plan = plan_columns([0, 1], [2, 3, 4, 5], [6, 7, 8, 9], 14)
    assert plan.n_completed == 4 and plan.n_dropped == 0
    assert plan.active.tolist() == [2, 3, 4, 5]
    assert plan.virtual.tolist() == [6, 7, 8, 9, 10, 11, 12, 13]


def test_shrinking_drops_the_highest_virtuals_and_leaves_the_rest_numbered_as_before():
    plan = plan_columns([0, 1], [2, 3, 4, 5], [6, 7, 8, 9], 8)
    assert plan.n_dropped == 2 and plan.n_completed == 0
    assert plan.keep.tolist() == [0, 1, 2, 3, 4, 5, 6, 7]
    assert plan.inactive.tolist() == [0, 1] and plan.active.tolist() == [2, 3, 4, 5]


def test_a_target_too_small_to_hold_the_calculation_is_refused():
    """⚠ Dropping an inactive or active orbital would turn the request into a different
    calculation wearing the same name, so it is refused rather than reinterpreted."""
    with pytest.raises(ValueError, match="never dropped"):
        plan_columns([0, 1], [2, 3, 4, 5], [6, 7, 8, 9], 4)


def test_a_dropped_virtual_below_the_active_space_renumbers_what_follows_it():
    """The general case: the partition need not be contiguous, so the surviving columns are
    renumbered and the plan is what says where each space went."""
    plan = plan_columns([0, 1], [8, 9], [2, 3, 4, 5, 6, 7], 8)
    assert plan.keep.tolist() == [0, 1, 2, 3, 4, 5, 8, 9]
    assert plan.active.tolist() == [6, 7]                 # was 8, 9
    assert plan.virtual.tolist() == [2, 3, 4, 5]


# --- the projection itself ------------------------------------------------------------------

def test_projecting_a_basis_onto_itself_is_the_identity(refs):
    """The one case whose answer is known without trusting anything in the module.

    Same basis on both sides: the projector is the identity on the span, the Gram matrix is
    already unit, and every scheme must hand back the orbitals it was given (up to the phase
    freedom inside a degenerate block, so the comparison is through the density).
    """
    ref = refs["small"]
    s_self = cross_overlap(ref.data.molecule, ref.data.molecule)
    c = ref.spinors_in_ao()
    for scheme in SCHEMES:
        pr = project_spinors(c, s_self, ref.orth, carry="all", scheme=scheme, report=False)
        assert np.allclose(pr.retained, 1.0, atol=1e-10)
        d0 = c @ c.conj().T
        d1 = pr.coeff @ pr.coeff.conj().T
        assert np.abs(d1 - d0).max() < 1e-10


def test_the_scalar_projector_is_the_same_map_as_the_spinor_one(refs):
    """⚠ The scalar SCF guess and the CASSCF basis projection are two consumers of **one**
    projector, and this is the assertion that keeps them one: expand a real scalar set into
    the spinor convention, project it both ways, and require the same orbitals. A second
    implementation of equation (1) would pass every test written against itself.
    """
    from kuiva.orth.project import project_scalar_orbitals

    src, tgt = refs["small"], refs["large"]
    s_cross = cross_overlap(src.data.molecule, tgt.data.molecule)
    c_scalar = np.asarray(src.data.mo_coeff, dtype=float)[:, :4]

    scalar, retained, _floor = project_scalar_orbitals(c_scalar, s_cross, tgt.orth)

    nao_s = int(src.data.nao)
    spinor = np.zeros((2 * nao_s, 2 * c_scalar.shape[1]), dtype=complex)
    spinor[:nao_s, 0::2] = c_scalar                    # unbarred partners, alpha block only
    spinor[nao_s:, 1::2] = c_scalar                    # barred partners, beta block
    pr = project_spinors(spinor, s_cross, tgt.orth, carry="all", scheme="symmetric",
                         repair_pairing=False, report=False)

    # The carried columns come first; everything after them is the completion the larger
    # basis adds, which the scalar routine has no counterpart for and does not build.
    nao_t, ncar = int(tgt.data.nao), 2 * c_scalar.shape[1]
    carried = np.asarray(pr.coeff)[:nao_t, :ncar][:, 0::2]
    assert np.abs(carried.real - scalar).max() < 1e-10
    assert np.abs(carried.imag).max() < 1e-12
    assert np.allclose(retained, np.asarray(pr.retained)[:ncar][0::2], atol=1e-10)


def test_the_scalar_projection_restores_orthonormality_in_the_target_metric(refs):
    """⚠ The failure this exists to prevent: a projector is not unitary, so without the
    Loewdin step the projected occupied set is not orthonormal and the density built from it
    has the wrong trace. Onto a *larger* basis the loss is small enough to be invisible in an
    energy and large enough to matter to a first Fock build.
    """
    from kuiva.orth.project import project_scalar_orbitals

    src, tgt = refs["large"], refs["small"]             # large -> small: a real loss of norm
    s_cross = cross_overlap(src.data.molecule, tgt.data.molecule)
    c = np.asarray(src.data.mo_coeff, dtype=float)[:, :3]
    projected, retained, _floor = project_scalar_orbitals(c, s_cross, tgt.orth)
    gram = projected.T @ np.asarray(tgt.data.s_ao) @ projected
    assert np.abs(gram - np.eye(3)).max() < 1e-12
    assert np.all(retained <= 1.0 + 1e-12)


def test_the_delivered_set_is_orthonormal_in_the_target_metric(refs):
    """⚠ In the *target's* metric. A projection that forgot the cross overlap, or applied the
    source overlap, still produces a set that is orthonormal in something."""
    pr = project_to_basis(refs["small"], refs["large"], carry="all", report=False)
    assert orthonormality(refs["large"], pr.coeff) < 1e-12
    pr = project_to_basis(refs["large"], refs["small"], carry="all", report=False)
    assert orthonormality(refs["small"], pr.coeff) < 1e-12


@pytest.mark.parametrize("scheme", SCHEMES)
def test_kramers_pairing_survives_every_scheme(refs, scheme):
    """⚠ Structural, not incidental: the projector is real and spin-independent so it commutes
    with ``T = -i sigma_y K``, and each scheme is a column transformation that preserves the
    self-duality of the Gram matrix. A scheme that mixed the two spin blocks differently would
    pass every other test in this file."""
    pr = project_to_basis(refs["small"], refs["large"], carry="all", scheme=scheme,
                          report=False)
    c = pr.coeff
    # ⚠ The pairing convention is an identity on the *coefficients* — ``c_2p+1 = T c_2p`` —
    # so it is checked as a difference and needs no metric. An overlap-based version would
    # need the target's own ``S`` and silently measures nothing without it.
    assert float(np.abs(c[:, 1::2] - time_reverse(c[:, ::2])).max()) < 1e-10


def test_the_completed_columns_are_the_targets_own_orbitals_in_energy_order(refs):
    """⚠ The property one signed comparison destroyed while changing nothing else.

    With ``carry="active"`` everything outside the active space is built in the target's own
    terms and assigned in ascending pseudo-canonical energy, so the inactive positions must
    come back holding the target SCF's *lowest* spinors — not an arbitrary basis of the space
    that happens to be left over.
    """
    small, large = refs["small"], refs["large"]
    space = casscf(small, character=("B", "p"), n_active=6, n_states=6,
                   report=False).active
    pr = project_to_basis(small, large, space=space, carry="active", report=False)
    ov = np.abs(pr.coeff.conj().T @ metric(large) @ large.spinors_in_ao())
    # each inactive column is one of the two lowest Kramers pairs of the target's own SCF
    for col in pr.plan.inactive:
        best = int(np.argmax(ov[col]))
        assert best < pr.plan.inactive.size
        assert ov[col, best] > 0.999


def test_the_retained_norm_is_the_norm_the_target_basis_holds(refs):
    """A projector's retained norm is ``<psi|P|psi>``, computable independently from the
    coefficients and the two overlaps. Going up it is near 1; going down it is not, and that
    is the number that says a large-to-small projection has thrown something away."""
    small, large = refs["small"], refs["large"]
    pr = project_to_basis(large, small, carry="all", report=False)
    c = large.spinors_in_ao()
    s_cross = cross_overlap(large.data.molecule, small.data.molecule)
    x = small.orth.x
    p = np.kron(np.eye(2), x.T @ s_cross)
    direct = np.einsum("ij,ij->j", np.conj(p @ c), p @ c).real
    assert np.allclose(pr.retained, direct[pr.plan.keep], atol=1e-12)
    assert pr.retained.min() < 0.9                       # the small basis does not hold it all


def test_the_principal_overlaps_are_invariant_to_a_rotation_inside_the_active_space(refs):
    """⚠ The diagnostic has to be a statement about the *space*, not about the columns.

    Rotating the source active orbitals among themselves is a redundant rotation that changes
    no CASSCF quantity; a fidelity that moved with it would be measuring the arbitrary basis
    the eigensolver returned.
    """
    small, large = refs["small"], refs["large"]
    space = casscf(small, character=("B", "p"), n_active=6, n_states=6,
                   report=False).active
    base = project_to_basis(small, large, space=space, carry="active", report=False)

    rng = np.random.default_rng(20260823)
    a = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))
    u = np.linalg.qr(a)[0]
    c = np.array(small.spinors_in_ao(), copy=True)
    c[:, space.spaces.active] = c[:, space.spaces.active] @ u
    turned = project_to_basis(small, large, c, space=space, carry="active", report=False)
    assert abs(turned.fidelity - base.fidelity) < 1e-10


def test_a_linearly_dependent_projected_block_is_refused_not_orthonormalized(refs):
    """⚠ Orthonormalizing a block whose Gram matrix is singular multiplies its noise by the
    inverse square root of the smallest eigenvalue — the same failure the working basis'
    linear-dependence threshold exists to prevent, so it is refused the same way."""
    ref = refs["small"]
    s_self = cross_overlap(ref.data.molecule, ref.data.molecule)
    c = np.array(ref.spinors_in_ao(), copy=True)
    c[:, 2:4] = c[:, 0:2]                               # a duplicated Kramers pair
    with pytest.raises(ValueError, match="linearly dependent"):
        project_spinors(c, s_self, ref.orth, carry="all", scheme="blocked", report=False)


def test_carrying_the_active_space_with_no_active_space_is_refused(refs):
    """⚠ Otherwise the default ``carry="active"`` with no partition would carry *nothing*,
    and hand back the target's own guess wearing the name of a projection — an orthonormal
    set of the right shape, containing no orbital of the source anywhere."""
    with pytest.raises(ValueError, match="no active space to carry"):
        project_to_basis(refs["small"], refs["large"], report=False)


def test_a_different_molecule_is_refused(refs):
    from kuiva.interface.pyscf_bridge import MoleculeSpec
    other = MoleculeSpec.from_molecule(kuiva.Molecule([("C", (0.0, 0.0, 0.0))], basis=LARGE))
    with pytest.raises(ValueError, match="same molecule"):
        cross_overlap(refs["small"].data.molecule, other)


def test_a_moved_geometry_warns_but_still_projects(refs, kuiva_caplog):
    """Carrying orbitals along a geometry scan is a legitimate and different thing to do; it
    warns rather than refusing, because doing it by accident gives a mediocre guess and not an
    obviously wrong one."""
    from dataclasses import replace
    moved = replace(refs["large"].data.molecule, atoms=(("B", (0.0, 0.0, 0.05)),))
    cross_overlap(refs["small"].data.molecule, moved)
    assert any("geometries differing" in r.message for r in kuiva_caplog.records)


# --- the calculation: a projected guess must change nothing but the cost --------------------

@pytest.mark.slow
def test_a_projected_casscf_reaches_the_same_answer_as_a_direct_one(refs):
    """⚠ The substantive claim. A starting guess is a guess: it may change how long the
    optimization takes and it may not change where it ends up. This asserts the energies and
    the spin-orbit spectrum, and deliberately not the iteration count: that is a measurement,
    it depends on the machine and on the optimizer mode, and a test that pinned it would fail
    for reasons that have nothing to do with the projection being right."""
    small, large = refs["small"], refs["large"]
    space = casscf(small, character=("B", "p"), n_active=6, n_states=6, report=False).active
    src = casscf(small, active=space, n_states=6, report=False)
    pr = project_to_basis(small, large, src.coeff, space=space, report=False)
    target_space = projected_active_space(pr, large, space.n_elec, "projected")

    got = casscf(large, active=target_space, n_states=6, coeff=pr.coeff, report=False)
    want = casscf(large, character=("B", "p"), n_active=6, n_states=6, report=False)
    assert abs(got.orbital.energy - want.orbital.energy) < E_TOL
    assert np.allclose(np.sort(got.ci.total_energies), np.sort(want.ci.total_energies),
                       atol=E_TOL)


# --- the class-API surface ------------------------------------------------------------------

@pytest.fixture(scope="module")
def stages():
    scf_s = ScalarSCF(boron(SMALL), memory_gb=8.0, screening="none").run()
    scf_l = ScalarSCF(boron(LARGE), memory_gb=8.0, screening="none").run()
    ref_s, ref_l = Reference(scf_s).run(), Reference(scf_l).run()
    cas_s = CASSCF(ref_s, character=("B", "p"), n_active=6, n_states=6, report=False).run()
    return dict(small=ref_s, large=ref_l, cas=cas_s)


def test_the_stage_inherits_the_active_space_from_the_projected_calculation(stages):
    cas = CASSCF(stages["large"], project_from=stages["cas"], n_states=6, report=False)
    assert cas.space.n_active == stages["cas"].active.n_active
    assert cas.space.n_elec == stages["cas"].active.n_elec
    assert "projected from" in cas.space.description


def test_restating_the_active_space_on_a_projection_is_refused(stages):
    """⚠ It would be resolved against *this* reference's guess orbitals, which is a different
    calculation from the one whose orbitals are being carried."""
    with pytest.raises(ValueError, match="comes across with the projected orbitals"):
        CASSCF(stages["large"], project_from=stages["cas"], character=("B", "p"),
               n_active=6, n_states=6)


def test_projecting_from_a_stage_on_the_same_reference_is_refused(stages):
    cas_l = CASSCF(stages["large"], character=("B", "p"), n_active=6, n_states=6,
                   report=False)
    with pytest.raises(ValueError, match="no basis to project between"):
        CASSCF(stages["large"], project_from=cas_l.run(), n_states=6)


def test_a_cheap_ci_upstream_with_a_projection_is_refused(stages):
    pre = CheapCI(stages["large"], character=("B", "p"), n_active=6).run()
    with pytest.raises(ValueError, match="needs a Reference upstream"):
        CASSCF(pre, project_from=stages["cas"], n_states=6)


def test_a_projection_and_a_restart_are_refused_together(stages, tmp_path):
    path = tmp_path / "x.h5"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="one or the other"):
        CASSCF(stages["large"], project_from=stages["cas"], restart=str(path), n_states=6)


def test_unknown_projection_options_fail_at_construction(stages):
    with pytest.raises(TypeError, match="unexpected option"):
        CASSCF(stages["large"], project_from=stages["cas"], n_states=6,
               projection=dict(nonsense=1))


def test_projection_options_without_a_projection_are_refused(stages):
    with pytest.raises(ValueError, match="configures project_from"):
        CASSCF(stages["large"], character=("B", "p"), n_active=6,
               projection=dict(scheme="symmetric"))


def test_projecting_a_plain_reference_needs_the_space_stated_once(stages):
    """A :class:`Reference` has no active space, so it is stated here — and resolved against
    the **source**, because those are the orbitals being carried."""
    cas = CASSCF(stages["large"], project_from=stages["small"], character=("B", "p"),
                 n_active=6, n_states=6, report=False)
    assert cas.space.n_active == 6
    with pytest.raises(ValueError, match="needs an active space"):
        CASSCF(stages["large"], project_from=stages["small"], n_states=6)


@pytest.mark.slow
def test_the_stage_converges_to_the_direct_answer(stages):
    got = CASSCF(stages["large"], project_from=stages["cas"], n_states=6, report=False).run()
    want = CASSCF(stages["large"], character=("B", "p"), n_active=6, n_states=6,
                  report=False).run()
    assert abs(got.energy - want.energy) < E_TOL
    assert np.allclose(got.energies, want.energies, atol=E_TOL)
    assert got.projection.fidelity > 0.99
