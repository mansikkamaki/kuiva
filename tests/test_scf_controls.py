"""The molecular SCF's convergence controls, orbital re-use and non-convergence policy.

Four groups, and each one tests a *mechanism* rather than only the observable it was found
through:

* **the named-value checks**, because PySCF substitutes ``minao`` silently for an
  ``init_guess`` it does not recognize and switching DIIS variant is a class assignment — an
  unvalidated passthrough runs something other than what was asked for and says nothing;
* **the starting guess**, whose failure mode is a density with the wrong number of electrons
  in it (a projector is not unitary) — asserted as ``Tr(D S) = N`` on both routes, not as a
  converged energy, which would pass with a badly normalized guess too;
* **stability**, on an atom whose ROHF converges to a saddle point 0.3 Eh above the solution
  next to it. That is the whole reason the analysis exists, and it is why the test asserts a
  *lower* energy rather than a flag;
* **non-convergence**, which now refuses.
"""
from dataclasses import replace

import numpy as np
import pytest

import kuiva
from kuiva.interface import pyscf_bridge as bridge
from kuiva.interface.pyscf_bridge import (SCF_DIIS_VARIANTS, SCF_INIT_GUESSES,
                                          STABILITY_MAX_FOLLOW, apply_scf_controls,
                                          validate_scf_controls)

BASE = dict(screening="none", with_soc=False, memory_gb=4.0)


def hf_molecule():
    return kuiva.Molecule([("H", (0.0, 0.0, 0.0)), ("F", (0.0, 0.0, 0.917))],
                          basis="x2c-SVPall-2c")


def water_cation():
    """An open-shell doublet: the restricted reference has a singly occupied orbital, which
    is what makes the alpha/beta split of a re-used guess a thing that can be got wrong."""
    return kuiva.Molecule([("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.757, 0.587)),
                           ("H", (0.0, -0.757, 0.587))],
                          basis="x2c-SVPall-2c", charge=1, spin=1)


# --- the named-value checks -----------------------------------------------------------------

@pytest.mark.parametrize("kwargs, expected", [
    (dict(level_shift=-0.1), "level_shift"),
    (dict(damp=1.0), "damp"),
    (dict(damp=-0.1), "damp"),
    (dict(diis="pulay"), "unknown diis"),
    (dict(diis_space=0), "diis_space"),
    (dict(diis_start_cycle=0), "diis_start_cycle"),
    (dict(init_guess="minoa"), "unknown init_guess"),
    (dict(stability="maybe"), "unknown stability"),
])
def test_bad_control_values_are_refused_by_name(kwargs, expected):
    with pytest.raises(ValueError, match=expected):
        validate_scf_controls(**kwargs)


def test_init_guess_chk_names_the_alternative():
    """⚠ The mechanism: PySCF's ``get_init_guess`` treats any ``chk*`` key as a checkpoint
    read and falls back to ``minao`` when the file is not there — silently. This front end
    owns no chkfile, so the request is refused and pointed at ``guess_from=``."""
    with pytest.raises(ValueError, match="guess_from"):
        validate_scf_controls(init_guess="chkfile")


def test_stability_boolean_is_refused_rather_than_mapped():
    """``True`` reads as either mode and the difference between them is a re-run of the SCF."""
    with pytest.raises(TypeError, match="check.*follow"):
        validate_scf_controls(stability=True)


def test_every_documented_name_validates():
    for name in SCF_INIT_GUESSES:
        assert validate_scf_controls(init_guess=name)["init_guess"] == name
    for name in SCF_DIIS_VARIANTS:
        assert validate_scf_controls(diis=name)["diis"] == name
    assert validate_scf_controls(diis=False)["diis"] is False
    assert validate_scf_controls(diis="off")["diis"] is False


def test_controls_reach_the_pyscf_object():
    """The knobs are passthroughs, so the mechanism to test is that they land."""
    from pyscf import gto, scf
    from pyscf.scf import diis as pyscf_diis

    mf = scf.RHF(gto.M(atom="He 0 0 0", basis="sto-3g", verbose=0))
    mf, note = apply_scf_controls(mf, level_shift=0.3, damp=0.4, init_guess="atom",
                                  diis="adiis", diis_space=12, diis_start_cycle=3)
    assert mf.level_shift == pytest.approx(0.3)
    assert mf.damp == pytest.approx(0.4)
    assert mf.init_guess == "atom"
    assert mf.DIIS is pyscf_diis.ADIIS
    assert mf.diis_space == 12
    assert mf.diis_start_cycle == 3
    for fragment in ("level_shift", "damp", "ADIIS", "diis_space", "init_guess"):
        assert fragment in note


def test_diis_off_switches_it_off():
    from pyscf import gto, scf

    mf, note = apply_scf_controls(scf.RHF(gto.M(atom="He 0 0 0", basis="sto-3g", verbose=0)),
                                  diis=False)
    assert mf.diis is None
    assert "DIIS off" in note


def test_defaults_touch_nothing_and_say_nothing():
    """A row per knob would put six defaults in every output file in the project."""
    from pyscf import gto, scf

    mf = scf.RHF(gto.M(atom="He 0 0 0", basis="sto-3g", verbose=0))
    before = (mf.level_shift, mf.damp, mf.DIIS, mf.diis_space, mf.diis_start_cycle)
    mf, note = apply_scf_controls(mf)
    assert note == ""
    assert (mf.level_shift, mf.damp, mf.DIIS, mf.diis_space, mf.diis_start_cycle) == before


def test_second_order_wraps_and_warns_about_the_knobs_it_ignores(kuiva_caplog):
    from pyscf import gto, scf

    mf = scf.RHF(gto.M(atom="He 0 0 0", basis="sto-3g", verbose=0))
    mf, note = apply_scf_controls(mf, level_shift=0.3, diis="adiis", second_order=True)
    assert hasattr(mf, "_scf")                      # the CIAH wrapper keeps the inner object
    assert "second-order" in note
    assert "level_shift" not in note                # it was not applied, so it is not reported
    assert any("does not use the first-order" in r.message for r in kuiva_caplog.records)


# --- the starting guess ---------------------------------------------------------------------

def _electrons(dm, s_ao):
    dm = np.asarray(dm)
    dm = dm if dm.ndim == 2 else dm.sum(axis=0)
    return float(np.einsum("ij,ji->", dm, s_ao))


@pytest.mark.parametrize("target_basis, route", [
    ("x2c-SVPall-2c", "reused directly"),
    ("x2c-TZVPall-2c", "projected"),
])
def test_reused_guess_carries_the_right_number_of_electrons(target_basis, route):
    """⚠ The mechanism, and the one that is invisible in a converged energy: a projector is
    not unitary, so raw projected orbitals lose norm and ``Tr(D S)`` comes out below ``N``.
    An SCF handed that guess builds its first Fock operator for a different system.
    """
    source = kuiva.ScalarSCF(hf_molecule(), **BASE).run()
    target = kuiva.Molecule([("H", (0.0, 0.0, 0.0)), ("F", (0.0, 0.0, 0.917))],
                            basis=target_basis)
    mol = bridge.build_mole(target)
    dm0, note = bridge._guess_density(
        source.data, mol, "rhf", bridge.ao_layout(mol),
        bridge.MoleculeSpec.from_molecule(target))
    assert route in note
    assert _electrons(dm0, mol.intor("int1e_ovlp")) == pytest.approx(10.0, abs=1e-9)


def test_same_basis_guess_reproduces_the_source_density_exactly():
    """The "same AO basis" route has to be the identity, or the potential-energy-surface use
    of it is silently doing a projection with no cross overlap to justify it."""
    source = kuiva.ScalarSCF(hf_molecule(), **BASE).run()
    mol = bridge.build_mole(hf_molecule())
    dm0, note = bridge._guess_density(source.data, mol, "rhf", bridge.ao_layout(mol),
                                      bridge.MoleculeSpec.from_molecule(hf_molecule()))
    c, occ = source.data.mo_coeff, source.data.mo_occ
    expected = (c * occ) @ c.T
    assert "same AO basis" in note
    assert np.allclose(dm0, expected, atol=1e-12)


def test_open_shell_guess_stays_spin_polarized_for_an_unrestricted_target():
    """⚠ The occupations are summed, never counted: an ROHF singly occupied orbital has
    ``mo_occ > 0`` while holding one electron, and counting puts an extra electron in."""
    source = kuiva.ScalarSCF(water_cation(), reference="rohf", **BASE).run()
    mol = bridge.build_mole(water_cation())
    dm0, _note = bridge._guess_density(source.data, mol, "uhf", bridge.ao_layout(mol),
                                       bridge.MoleculeSpec.from_molecule(water_cation()))
    assert np.asarray(dm0).shape == (2, mol.nao, mol.nao)
    s_ao = mol.intor("int1e_ovlp")
    assert np.einsum("ij,ji->", dm0[0], s_ao) == pytest.approx(5.0, abs=1e-9)
    assert np.einsum("ij,ji->", dm0[1], s_ao) == pytest.approx(4.0, abs=1e-9)
    assert not np.allclose(dm0[0], dm0[1])


def test_an_unrestricted_source_carries_its_two_orbital_sets():
    """⚠ ``mo_sets()`` rather than a branch on ``mo_coeff.ndim``: a UHF source has two sets and
    two occupation vectors, and reading a ``(2, nao, nmo)`` array as ``(nao, nmo)`` of a
    different basis is the standard way to get this wrong."""
    source = kuiva.ScalarSCF(water_cation(), reference="uhf", **BASE).run()
    assert source.data.unrestricted
    mol = bridge.build_mole(water_cation())
    dm0, _note = bridge._guess_density(source.data, mol, "uhf", bridge.ao_layout(mol),
                                       bridge.MoleculeSpec.from_molecule(water_cation()))
    s_ao = mol.intor("int1e_ovlp")
    assert np.einsum("ij,ji->", dm0[0], s_ao) == pytest.approx(5.0, abs=1e-9)
    assert np.einsum("ij,ji->", dm0[1], s_ao) == pytest.approx(4.0, abs=1e-9)
    c_a, c_b = source.data.mo_sets()
    expected_a = (c_a * source.data.mo_occ[0]) @ c_a.T
    assert np.allclose(dm0[0], expected_a, atol=1e-12)


def test_stability_off_is_accepted_and_measures_nothing():
    """``False`` normalizes to ``None`` rather than reaching the run as a string operation on
    a bool — the shape of bug a knob with three falsy spellings invites."""
    assert validate_scf_controls(stability=False)["stability"] is None
    scf = kuiva.ScalarSCF(hf_molecule(), stability=False, **BASE).run()
    assert scf.stable is None


def test_projected_guess_reaches_the_same_solution_as_a_cold_start():
    """A guess may change the cost and may not change the answer."""
    small = kuiva.ScalarSCF(hf_molecule(), **BASE).run()
    big = kuiva.Molecule([("H", (0.0, 0.0, 0.0)), ("F", (0.0, 0.0, 0.917))],
                         basis="x2c-TZVPall-2c")
    cold = kuiva.ScalarSCF(big, **BASE).run()
    warm = kuiva.ScalarSCF(big, guess_from=small, **BASE).run()
    assert warm.energy == pytest.approx(cold.energy, abs=1e-9)


def test_the_ways_to_start_an_scf_are_refused_together():
    """⚠ There are three of them now — ``guess_from=``, ``init_guess=`` and
    ``broken_symmetry=`` — and they are three statements about the same thing, so any two are
    a contradiction rather than a preference."""
    source = kuiva.ScalarSCF(hf_molecule(), **BASE).run()
    with pytest.raises(ValueError, match="statements about where the SCF"):
        kuiva.ScalarSCF(hf_molecule(), guess_from=source, init_guess="atom", **BASE).run()


def test_guess_from_a_different_molecule_is_refused():
    source = kuiva.ScalarSCF(hf_molecule(), **BASE).run()
    other = kuiva.Molecule([("H", (0.0, 0.0, 0.0)), ("Cl", (0.0, 0.0, 1.27))],
                           basis="x2c-TZVPall-2c")
    with pytest.raises(ValueError, match="same molecule"):
        kuiva.ScalarSCF(other, guess_from=source, **BASE).run()


def test_projection_without_a_molecule_spec_is_refused_and_says_why():
    """A container built by hand carries no ``MoleculeSpec``, so its basis cannot be rebuilt
    and the cross overlap cannot be formed. The same-basis route still works without one."""
    source = kuiva.ScalarSCF(hf_molecule(), **BASE).run()
    stripped = replace(source.data, molecule=None)
    big = kuiva.Molecule([("H", (0.0, 0.0, 0.0)), ("F", (0.0, 0.0, 0.917))],
                         basis="x2c-TZVPall-2c")
    with pytest.raises(ValueError, match="carries no MoleculeSpec"):
        kuiva.ScalarSCF(big, guess_from=stripped, **BASE).run()


def test_stage_validates_the_guess_eagerly():
    unrun = kuiva.ScalarSCF(hf_molecule(), **BASE)
    with pytest.raises(ValueError, match="has not been run"):
        kuiva.ScalarSCF(hf_molecule(), guess_from=unrun, **BASE)
    with pytest.raises(TypeError, match="guess_from"):
        kuiva.ScalarSCF(hf_molecule(), guess_from="orbitals.h5", **BASE)


def test_stage_validates_the_controls_eagerly():
    """Construction, not an hour into the run — which for a lanthanide is the whole point."""
    with pytest.raises(ValueError, match="unknown diis"):
        kuiva.ScalarSCF(hf_molecule(), diis="pulay", **BASE)
    with pytest.raises(ValueError, match="level_shift"):
        kuiva.ScalarSCF(hf_molecule(), level_shift=-1.0, **BASE)


# --- stability -------------------------------------------------------------------------------

def test_stability_follow_finds_the_lower_solution():
    """⚠ The mechanism, on the cheapest system that shows it: the Ni atom's ROHF converges,
    reports every diagnostic clean, and sits on a **saddle point** of the SCF energy 0.30 Eh
    above the solution one rotation away. Nothing else in the front end can see that — a
    converged flag, a gradient norm and a plausible energy are all present.
    """
    ni = kuiva.Molecule([("Ni", (0.0, 0.0, 0.0))], basis="x2c-SVPall-2c", spin=2)
    saddle = kuiva.ScalarSCF(ni, max_cycle=100, stability="check", **BASE).run()
    lower = kuiva.ScalarSCF(ni, max_cycle=100, stability="follow", **BASE).run()
    assert saddle.converged and lower.converged
    assert lower.energy < saddle.energy - 0.1


def test_stability_check_warns_without_re_solving(kuiva_caplog):
    ni = kuiva.Molecule([("Ni", (0.0, 0.0, 0.0))], basis="x2c-SVPall-2c", spin=2)
    scf = kuiva.ScalarSCF(ni, max_cycle=100, stability="check", **BASE).run()
    assert any("internally UNSTABLE" in r.message for r in kuiva_caplog.records)
    assert scf.energy == pytest.approx(-1518.50139, abs=1e-3)


def test_stability_follow_stops_after_the_cap(kuiva_caplog):
    """A follow is a bounded loop: an SCF still unstable after three re-solves is a statement
    about the system, and the honest report is the warning rather than a fourth attempt."""
    calls = {"stability": 0, "kernel": 0}

    class _Stub:
        mo_occ = np.array([2.0, 0.0])
        converged = True

        def stability(self, internal=True, external=False, return_status=False):
            calls["stability"] += 1
            return np.eye(2), None, False, None            # never stable

        def make_rdm1(self, mo, occ):
            return np.eye(2)

        def kernel(self, dm0=None):
            calls["kernel"] += 1
            return -1.0

    mf, e_scf, stable, n_follow = bridge._stability_follow(_Stub(), "follow")
    assert not stable
    assert n_follow == STABILITY_MAX_FOLLOW
    assert calls["kernel"] == STABILITY_MAX_FOLLOW
    assert calls["stability"] == STABILITY_MAX_FOLLOW + 1
    assert e_scf == -1.0
    assert any("internally UNSTABLE" in r.message for r in kuiva_caplog.records)


def test_stability_check_never_re_solves():
    """``"check"`` measures and reports; re-solving is what ``"follow"`` is for."""
    calls = {"kernel": 0}

    class _Stub:
        mo_occ = np.array([2.0, 0.0])

        def stability(self, internal=True, external=False, return_status=False):
            return np.eye(2), None, False, None

        def kernel(self, dm0=None):                        # pragma: no cover - must not run
            calls["kernel"] += 1
            return -1.0

    _mf, e_scf, stable, n_follow = bridge._stability_follow(_Stub(), "check")
    assert (e_scf, stable, n_follow) == (None, False, 0)
    assert calls["kernel"] == 0


# --- non-convergence ---------------------------------------------------------------------------

def test_unconverged_scf_refuses_and_names_the_levers():
    with pytest.raises(RuntimeError, match="did not converge") as excinfo:
        kuiva.ScalarSCF(hf_molecule(), max_cycle=2, **BASE).run()
    message = str(excinfo.value)
    for lever in ("level_shift", "adiis", "second_order", "max_cycle", "guess_from",
                  "allow_unconverged_scf"):
        assert lever in message


def test_allow_unconverged_scf_warns_and_continues(kuiva_caplog):
    scf = kuiva.ScalarSCF(hf_molecule(), max_cycle=2, allow_unconverged_scf=True,
                          **BASE).run()
    assert not scf.converged
    assert any("did not converge" in r.message and r.levelname == "WARNING"
               for r in kuiva_caplog.records)


def test_second_order_converges_what_the_diis_iteration_cannot():
    """The stage's reason for existing, on a case that genuinely fails: CrO from a bare
    one-electron guess oscillates, and it is **not** a budget artefact — measured, the DIIS
    iteration is still unconverged at 100 cycles, wandering around -1124.140 Eh — while the
    CIAH solver converges inside 40 to a *lower* -1124.156078 Eh.
    """
    cro = kuiva.Molecule([("Cr", (0.0, 0.0, 0.0)), ("O", (0.0, 0.0, 1.62))],
                         basis="x2c-SVPall-2c", spin=2)
    with pytest.raises(RuntimeError, match="did not converge"):
        kuiva.ScalarSCF(cro, init_guess="1e", max_cycle=40, **BASE).run()
    rescued = kuiva.ScalarSCF(cro, init_guess="1e", max_cycle=40, second_order=True,
                              **BASE).run()
    assert rescued.converged
    assert rescued.energy == pytest.approx(-1124.156078, abs=1e-4)
