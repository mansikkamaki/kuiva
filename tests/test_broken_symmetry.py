"""The broken-symmetry guess: the lever that did not exist, and the two ways it can lie.

What this file is for. A broken-symmetry solution is a *claim* about where the spin sits, and
two things have to be true before it is worth anything: the determinant must actually be
polarized (``<S^2>`` between the low-spin and high-spin values, not at the low-spin one), and
the polarization must be **where it was asked to be** (spin populations with the assigned
signs). A run that converged back to the symmetric solution has the same shape as one that
worked — same reference name, same orbital count, an energy that looks fine — so both are
asserted, and so is the premise: an unrestricted SCF started the ordinary way stays symmetric.

Two systems, both seconds: stretched H2, where the exact answer is known and the coupled
dimer is a textbook one, and two bare Ti(3+) ions, which is the polynuclear d^1 motif the
tensor-network layer exists for with everything but the coupling stripped away.
"""
import numpy as np
import pytest

from kuiva.interface import Molecule
from kuiva.interface.broken_symmetry import broken_symmetry_density, resolve_spins
from kuiva.interface.pyscf_bridge import run_scalar_x2c
from kuiva.props.population import scalar_spin_populations


def _h2(r=3.0):
    return Molecule.from_xyz_string("H 0 0 0\nH 0 0 {}".format(r), basis="x2c-SVPall-2c")


@pytest.fixture(scope="module")
def symmetric_h2():
    """A plain unrestricted SCF on stretched H2 — the state of affairs before this existed."""
    return run_scalar_x2c(_h2(), reference="uhf", screening="none")


@pytest.fixture(scope="module")
def broken_h2():
    return run_scalar_x2c(_h2(), reference="uhf", screening="none",
                          broken_symmetry={1: +1, 2: -1})


# --- the premise ---------------------------------------------------------------------------

def test_an_unrestricted_scf_started_the_ordinary_way_stays_symmetric(symmetric_h2):
    """⚠ The whole reason a guess has to be built. The closed-shell density is a *stationary*
    point, so nothing in the iteration pushes off it: the user asks for UHF on a dissociating
    bond, gets the restricted answer, and has no way to tell from the output that they did."""
    assert symmetric_h2.unrestricted                      # it really is a UHF
    assert abs(symmetric_h2.s2_deviation) < 1e-6          # and it is not polarized at all
    np.testing.assert_allclose(scalar_spin_populations(symmetric_h2), 0.0, atol=1e-6)


# --- what the guess buys -------------------------------------------------------------------

def test_the_broken_symmetry_solution_is_polarized_and_lower(broken_h2, symmetric_h2):
    """The three statements together, because any two of them can be true of a wrong answer:
    polarized, polarized *the way it was asked*, and lower than the symmetric solution it
    would otherwise have returned."""
    s2 = broken_h2.s2_deviation                            # Ms = 0, so the exact value is 0
    assert 0.5 < s2 < 2.0                                  # between the low-spin and high-spin
    spins = scalar_spin_populations(broken_h2)
    assert spins[0] > 0.9 and spins[1] < -0.9
    assert broken_h2.e_scf < symmetric_h2.e_scf - 0.1      # measured 0.17 Eh on this geometry


def test_the_guess_is_built_from_the_high_spin_solution_it_reports(broken_h2):
    """The recipe, checked rather than described: the magnetic orbitals come from the
    high-spin state, and the energy the guess reports for it is that state's own."""
    from kuiva.interface.pyscf_bridge import build_mole
    from kuiva.interface.api import Molecule as _M

    mol = build_mole(_h2())
    layout = broken_h2.ao_layout
    guess = broken_symmetry_density(mol, {1: +1, 2: -1}, layout, report=False)
    triplet = run_scalar_x2c(Molecule.from_xyz_string(
        "H 0 0 0\nH 0 0 3.0", basis="x2c-SVPall-2c", spin=2), reference="uhf",
        screening="none")
    assert guess.e_high_spin == pytest.approx(triplet.e_scf, abs=1e-8)
    assert guess.s2_high_spin == pytest.approx(2.0, abs=1e-2)
    assert guess.n_flipped == 1 and guess.weakest > 0.99
    assert guess.dm0.shape == (2, mol.nao, mol.nao)


def test_a_metal_dimer_polarizes_the_way_it_was_told_to():
    """The polynuclear d^1 motif, stripped to two bare Ti(3+) ions: the case the tensor-network
    layer exists for, and the one where "which centre" cannot be read off the canonical
    orbitals at all (they are the symmetric and antisymmetric combinations of two d shells)."""
    mol = Molecule([("Ti", (0.0, 0.0, 0.0)), ("Ti", (4.0, 0.0, 0.0))],
                   basis="x2c-SVPall-2c", charge=6, spin=0)
    broken = run_scalar_x2c(mol, reference="uhf", screening="none",
                            broken_symmetry={"Ti1": +1, "Ti2": -1})
    symmetric = run_scalar_x2c(mol, reference="uhf", screening="none")
    assert 0.5 < broken.s2_deviation < 2.0
    spins = scalar_spin_populations(broken)
    assert spins[0] > 0.9 and spins[1] < -0.9
    assert broken.e_scf < symmetric.e_scf
    np.testing.assert_allclose(scalar_spin_populations(symmetric), 0.0, atol=1e-4)


def test_the_assignment_can_be_reversed_and_the_solution_follows_it():
    """⚠ Swapping the signs is a *different state* with the same energy and the same <S^2>,
    which is exactly why the sign check exists: neither of the other two diagnostics can tell
    these two runs apart."""
    up = run_scalar_x2c(_h2(), reference="uhf", screening="none",
                        broken_symmetry={1: +1, 2: -1})
    down = run_scalar_x2c(_h2(), reference="uhf", screening="none",
                          broken_symmetry={1: -1, 2: +1})
    assert up.e_scf == pytest.approx(down.e_scf, abs=1e-9)
    assert up.s2_deviation == pytest.approx(down.s2_deviation, abs=1e-9)
    np.testing.assert_allclose(scalar_spin_populations(up),
                               -scalar_spin_populations(down), atol=1e-6)


# --- the refusals --------------------------------------------------------------------------

def test_the_assignment_must_agree_with_the_molecule_s_own_spin():
    """Two statements of the same thing, so they are made to agree rather than one silently
    winning: an assignment leaving one net unpaired electron on a molecule built with
    spin = 0 is a contradiction, not a preference."""
    with pytest.raises(ValueError, match="2 Ms"):
        run_scalar_x2c(_h2(), reference="uhf", screening="none",
                       broken_symmetry={1: +2, 2: -1})


def test_one_sign_everywhere_is_the_high_spin_state_and_is_refused():
    mol = Molecule.from_xyz_string("H 0 0 0\nH 0 0 3.0", basis="x2c-SVPall-2c", spin=2)
    with pytest.raises(ValueError, match="high-spin state"):
        run_scalar_x2c(mol, reference="uhf", screening="none",
                       broken_symmetry={1: +1, 2: +1})


def test_a_bare_element_symbol_on_a_homonuclear_dimer_cannot_say_which_centre():
    """``{"H": +1}`` assigns *every* hydrogen the same sign — the symmetric guess wearing a
    fragment label. Refused by the same rule, since it is the same mistake."""
    with pytest.raises(ValueError, match="high-spin state|2 Ms"):
        run_scalar_x2c(_h2(), reference="uhf", screening="none", broken_symmetry={"H": +1})


def test_a_restricted_reference_has_nowhere_to_put_the_polarization():
    with pytest.raises(ValueError, match="uhf"):
        run_scalar_x2c(_h2(), reference="rhf", screening="none",
                       broken_symmetry={1: +1, 2: -1})


def test_the_three_ways_to_start_an_scf_are_mutually_exclusive(symmetric_h2):
    for extra in ({"init_guess": "atom"}, {"guess_from": symmetric_h2}):
        with pytest.raises(ValueError, match="only one of them can be true"):
            run_scalar_x2c(_h2(), reference="uhf", screening="none",
                           broken_symmetry={1: +1, 2: -1}, **extra)


def test_an_assignment_the_state_does_not_have_is_refused():
    """Asking for two unpaired electrons on each hydrogen: the high-spin solution would need
    four singly occupied orbitals and the molecule has two electrons. The count is a property
    of the state, not of the request, and the refusal says so before an SCF is started."""
    mol = Molecule.from_xyz_string("H 0 0 0\nH 0 0 3.0", basis="x2c-SVPall-2c")
    with pytest.raises(ValueError, match="no state of this molecule has"):
        run_scalar_x2c(mol, reference="uhf", screening="none",
                       broken_symmetry={1: +2, 2: -2})


def test_delocalized_magnetic_orbitals_are_refused_with_the_table():
    """⚠ The silent failure this guards: flipping an orbital that is half on the other centre
    produces a density that is not the requested pattern, and the SCF then converges back to
    the symmetric solution with nothing having gone visibly wrong. At the equilibrium bond
    length the two 1s orbitals still localize; demanding near-perfection is the reproducible
    way to reach the refusal."""
    with pytest.raises(ValueError, match="min_population"):
        run_scalar_x2c(_h2(0.74), reference="uhf", screening="none",
                       broken_symmetry={1: +1, 2: -1}, bs_min_population=0.999)


def test_the_spin_assignment_addressing_is_the_project_s_own():
    """Element, label and 1-based number all resolve, because "which atom" has one meaning
    across per-atom bases, reference configurations and this."""
    assert resolve_spins({"Ti1": +1, "Ti2": -1}, ["Ti", "Ti"]) == [1, -1]
    assert resolve_spins({1: +1, 2: -1}, ["Ti", "Ti"]) == [1, -1]
    assert resolve_spins({"Ti": +1, "O": -1}, ["Ti", "O"]) == [1, -1]
    with pytest.raises(ValueError, match="no unpaired electrons"):
        resolve_spins({"Ti1": 0, "Ti2": 0}, ["Ti", "Ti"])


def test_a_restricted_container_has_no_spin_populations(symmetric_h2):
    rhf = run_scalar_x2c(Molecule.from_xyz_string("H 0 0 0\nH 0 0 0.74",
                                                  basis="x2c-SVPall-2c"),
                         reference="rhf", screening="none")
    with pytest.raises(ValueError, match="unrestricted"):
        scalar_spin_populations(rhf)
