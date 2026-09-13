"""Tier 0/1: which atoms the automatic selection calls magnetic centres.

The two conditions and the two ways they fail
---------------------------------------------
An atom is a centre for ``l`` when a frontier Kramers pair carries most of its population on
``(atom, l)`` **and** the atom's own free-atom reference has an open shell of that ``l``.
Each condition is here because dropping it produces a well-formed calculation about nothing,
and both failures are on real systems of this suite rather than imagined:

* ``zn2p`` -- Zn(2+)'s **filled** 3d sits in the frontier window and carries 1.000 of five
  pairs. Condition 1 alone makes it a centre; the ``d10`` reference is what keeps a closed
  shell out of an active space.
* ``cecl3`` -- the cerium's frontier orbitals carry 0.765 of their population on its **5d**
  and 0.919 on the 4f. Condition 1 alone cannot say which shell the physics is in; the
  Ce(3+) reference has ``d`` closed and ``f`` open, and that is what chooses.

The pooled case is ``ti2cl6``, where neither titanium clears the threshold alone (0.456) and
the two together do (0.913) -- the normal situation for equivalent centres, since the
canonical orbitals are combinations over both.

⚠ The refusals are tested against **stubs** where an SCF would prove nothing: what an
unrestricted reference or a missing atomic reference does is decided before any number is
computed, and a test that paid for an SCF to find that out would be a slow test of a
conditional.
"""
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "generate"))

import systems as sysdef                                                    # noqa: E402

from kuiva.autocas import centres as ctr                                    # noqa: E402
from kuiva.autocas import multiplets as mult                                # noqa: E402
from kuiva.interface import api                                             # noqa: E402


def reference_for(key):
    """A finished spinor reference for a committed system, with the free-atom orbitals.

    ``screening="none"``: nothing here is a statement about spin-orbit coupling, and the
    suite may not depend on a warm X2CAMF cache. ``atomic_reference=True`` is the whole point
    -- it is the second of the two conditions.
    """
    system = sysdef.get(key)
    molecule = api.Molecule(atoms=system.atoms, basis=system.basis, charge=system.charge,
                            spin=system.spin)
    data = api.scalar_x2c_reference(molecule, screening="none", memory_gb=8.0,
                                    atomic_reference=True, **system.scf_options)
    return api.spinor_reference(data, memory_gb=8.0)


@pytest.fixture(scope="module")
def ticl3():
    return reference_for("ticl3")


@pytest.fixture(scope="module")
def zn2p():
    return reference_for("zn2p")


# --- detection on a single centre ------------------------------------------------------------

def test_the_titanium_of_ticl3_is_a_d_centre_of_five_pairs(ticl3):
    detection = ctr.detect_centres(ticl3, report=False)
    assert len(detection.centres) == 1
    centre = detection.centres[0]
    assert centre.atoms == (0,) and centre.l == 2 and not centre.pooled
    assert centre.n_pairs == 5
    assert centre.population > 0.9
    assert centre.labels == ("1 Ti",)


def test_the_chlorines_are_not_centres(ticl3):
    """⚠ Their ``p`` shell is open in the free-atom reference (Cl is ``p5``), and they are
    still not centres: an open ``p`` shell is a main-group radical, which is a different
    target class, not a magnetic centre."""
    detection = ctr.detect_centres(ticl3, report=False)
    assert all(centre.l in ctr.CENTRE_ANGULAR_MOMENTA for centre in detection.centres)
    assert [c.labels for c in detection.centres] == [("1 Ti",)]


def test_the_reference_state_is_read_off_the_free_atom_solution(ticl3):
    """Per-channel electron counts, not a parse of the configuration label: a label is
    provenance and some of them ("Madelung filling") are not notation at all."""
    channels = ctr.reference_channels(ticl3.ao_layout, ticl3.data.atomic_reference, 0)
    assert np.allclose(channels[:4], [8.0, 12.0, 2.0, 0.0])    # neutral Ti: [Ar] 3d2 4s2
    # ⚠ ``approx``, not ``==``: these are the SUM of an atomic SCF's occupation-weighted AO
    # weights, so the last bits move with the reduction order and differ between runs on a
    # threaded BLAS. What is under test is that the channel holds eleven electrons.
    assert ctr.reference_channels(ticl3.ao_layout, ticl3.data.atomic_reference,
                                  1)[1] == pytest.approx(11.0)


def test_the_detected_centre_gives_the_floor_of_its_reference_state(ticl3):
    """⚠ And it is not the molecule's count: neutral Ti is ``d2``, TiCl3's titanium is
    ``d1``. That is exactly why the *active space*'s electron count is measured from the
    orbitals and the reference state is only allowed to cross-check it."""
    centre = ctr.detect_centres(ticl3, report=False).centres[0]
    assert centre.open_electrons == 2
    assert mult.shell_floor(centre.l, centre.open_electrons) == 3
    assert not centre.stated and not centre.trusted_state


# --- the closed shell ------------------------------------------------------------------------

def test_a_filled_3d_in_the_frontier_window_is_not_a_centre(zn2p):
    """⚠ The measurement that makes condition 2 load-bearing: Zn(2+)'s filled 3d carries the
    whole of five frontier pairs, so the population condition alone would make it a centre --
    and a filled shell as an active space is a calculation with plausible numbers and no
    physics in it."""
    detection = ctr.detect_centres(zn2p, report=False)
    assert detection.centres == ()
    population = detection.fractions[(0, 2)]
    assert float(np.max(population)) > 0.99, "the d character really is there"


def test_asking_for_the_closed_shell_by_name_is_refused_with_its_reference_state(zn2p):
    with pytest.raises(ValueError, match="CLOSED d shell"):
        ctr.shell_centre(zn2p, "Zn", "d")


def test_asking_for_a_shell_the_reference_has_no_open_channel_for_is_refused(zn2p):
    with pytest.raises(ValueError, match="no open d or f shell"):
        ctr.shell_centre(zn2p, "Zn")


# --- named shells ----------------------------------------------------------------------------

def test_a_named_shell_resolves_to_the_same_centre_as_the_detection(ticl3):
    detected = ctr.detect_centres(ticl3, report=False).centres[0]
    named = ctr.shell_centre(ticl3, "Ti")
    assert (named.atoms, named.l, named.n_pairs) == (detected.atoms, detected.l,
                                                     detected.n_pairs)
    assert ctr.shell_centre(ticl3, "Ti", "d") == named
    assert ctr.shell_centre(ticl3, 1, 2) == named


def test_the_shell_of_an_l_the_atom_has_no_open_channel_for_is_refused(ticl3):
    with pytest.raises(ValueError, match="CLOSED f shell|no l = 3"):
        ctr.shell_centre(ticl3, "Ti", "f")


def test_a_named_shell_whose_character_is_absent_warns_but_is_taken(ticl3, kuiva_caplog):
    """A shell asked for by name is the user's statement; what the code owes them is the
    number that says the orbitals are not where they thought."""
    centre = ctr.shell_centre(ticl3, "Ti", "d", threshold=0.99)
    assert centre.n_pairs == 5
    assert any("centre threshold" in r.message for r in kuiva_caplog.records)


# --- the refusals that need no SCF -----------------------------------------------------------

def _stub(*, unrestricted=False, atomic_reference=object()):
    return types.SimpleNamespace(
        data=types.SimpleNamespace(unrestricted=unrestricted,
                                   atomic_reference=atomic_reference))


def test_an_unrestricted_reference_is_refused_and_names_the_knob():
    """⚠ Its spinors are orthonormal but not Kramers paired, so a pair population is not a
    population of one orbital and the shell construction cannot fold them onto pairs."""
    with pytest.raises(ValueError, match="Kramers paired"):
        ctr.detect_centres(_stub(unrestricted=True))
    with pytest.raises(ValueError, match="reference='rohf'"):
        ctr.detect_centres(_stub(unrestricted=True))


def test_a_missing_atomic_reference_names_the_knob_and_refuses_a_fallback():
    """⚠ And it says why there is no fallback: a character threshold would be a *second*
    construction of "the shell", i.e. a second definition of it."""
    with pytest.raises(ValueError, match="atomic_reference=True"):
        ctr.detect_centres(_stub(atomic_reference=None))
    with pytest.raises(ValueError, match="two definitions of it"):
        ctr.detect_centres(_stub(atomic_reference=None))


# --- the configuration cross-check -----------------------------------------------------------

def _centre(*, stated, channels=(8.0, 12.0, 1.0, 0.0)):
    return ctr.Centre(atoms=(0,), l=2, labels=("1 Ti",), population=0.9, pooled=False,
                      reference_state="Ti(3+) [Ar]3d1", reference_channels=channels,
                      stated=stated, trusted_state=stated)


def test_nothing_is_checked_where_no_configuration_was_stated():
    """⚠ The default is not evidence: outside the f block it is the *neutral atom*, so "the
    neutral titanium disagrees with the ion" is not information."""
    assert ctr.check_configuration(_centre(stated=False), 3.0) is None


def test_a_stated_configuration_that_disagrees_by_more_than_one_electron_warns(kuiva_caplog):
    centre = _centre(stated=True)
    assert centre.open_electrons == 1
    assert ctr.check_configuration(centre, 1.0) is None        # agrees
    assert ctr.check_configuration(centre, 2.0) is None        # one electron: quiet
    text = ctr.check_configuration(centre, 3.0)
    assert text and "1" in text and "3.0" in text
    assert any("not the calculation that was meant" in r.message
               for r in kuiva_caplog.records)


def test_the_open_shell_count_is_the_channel_modulo_a_filled_shell():
    """A 5d metal's ``d`` channel holds its filled 3d and 4d too, so the *open* count is the
    channel total modulo ``4l+2`` -- which is what a reference state's shell occupation is."""
    centre = _centre(stated=True, channels=(10.0, 24.0, 26.0, 0.0))     # neutral Os: 5d6
    assert centre.reference_electrons == 26.0 and centre.open_electrons == 6


# --- the slow systems -------------------------------------------------------------------------

@pytest.mark.slow
def test_the_cerium_is_an_f_centre_and_not_the_d_one_its_populations_also_show():
    """⚠ Both channels clear the population condition; the reference state is what chooses.

    Measured: 0.919 on the Ce 4f and 0.765 on the Ce 5d. The Ce(3+) reference has 20
    electrons in its ``d`` channel (filled 3d and 4d) and one in ``f``.
    """
    reference = reference_for("cecl3")
    detection = ctr.detect_centres(reference, report=False)
    assert [(c.labels, c.l, c.n_pairs) for c in detection.centres] == [(("1 Ce",), 3, 7)]
    assert float(np.max(detection.fractions[(0, 2)])) > 0.5, "the 5d character is there too"
    centre = detection.centres[0]
    assert centre.open_electrons == 1 and centre.trusted_state, "the f-block default is M(3+)"
    assert mult.shell_floor(centre.l, centre.open_electrons) == 6          # 2F5/2


@pytest.mark.slow
def test_two_equivalent_titaniums_are_pooled_into_one_centre():
    """⚠ Neither clears the threshold alone (0.456 measured) and the two together do (0.913):
    the canonical orbitals of equivalent centres are combinations over both, so "which
    titanium" is not a question the reference orbitals answer."""
    reference = reference_for("ti2cl6")
    detection = ctr.detect_centres(reference, report=False)
    assert len(detection.centres) == 1
    centre = detection.centres[0]
    assert centre.atoms == (0, 1) and centre.pooled
    assert centre.n_pairs == 10, "2l+1 pairs per atom"
    assert centre.population > 0.8
    assert float(np.max(detection.fractions[(0, 2)])) < ctr.DEFAULT_CENTRE_THRESHOLD


def test_the_iron_of_fecl2_is_a_d_centre_with_six_reference_electrons():
    """In the default suite (2.5 s) although the system is marked slow: what is slow about
    ``fecl2`` is its CASSCF, and its high-spin ROHF reference is the only committed one whose
    shell is mostly *occupied*."""
    reference = reference_for("fecl2")
    centre = ctr.detect_centres(reference, report=False).centres[0]
    assert (centre.labels, centre.l, centre.n_pairs) == (("1 Fe",), 2, 5)
    assert centre.open_electrons == 6                    # neutral Fe is d6 as well as Fe(2+)
    assert mult.shell_floor(centre.l, centre.open_electrons) == 5          # 5D, spin floor
