"""Fragment localization: the site partition, and the two things it must never move.

What this file is for. Localization exists to answer *which centre* an orbital belongs to,
and it earns that answer by changing nothing else: the rotation is active-active, so a CASCI
energy is invariant to machine precision, and the Kramers pairing the whole active-space
convention is addressed in has to survive it. Both are asserted here as identities, not as
tolerances.

The systems are deliberately trivial — stretched H2, where the canonical orbitals are exactly
the symmetric and antisymmetric combinations, and two Ti(3+) ions far enough apart that the
answer is known before the calculation. A localizer that cannot separate those cannot separate
anything, and both run in under a second.
"""
import numpy as np
import pytest

from kuiva.interface import Molecule
from kuiva.interface.api import (active_space_for, casci, localize_active_space,
                                 spinor_reference)
from kuiva.mcscf.localize import fragment_populations, localize
from kuiva.spinor.expand import time_reverse


@pytest.fixture(scope="module")
def h2():
    """Stretched H2: two 1s orbitals, and the SCF returns them as sigma_g and sigma_u."""
    mol = Molecule.from_xyz_string("H 0 0 0\nH 0 0 3.0", basis="x2c-SVPall-2c")
    return spinor_reference(mol, screening="none")


@pytest.fixture(scope="module")
def h2_space(h2):
    return active_space_for(h2, character=([0, 1], "s"), n_active=4, n_active_elec=2)


# --- what the canonical orbitals cannot say ------------------------------------------------

def test_the_canonical_orbitals_are_half_on_each_centre(h2, h2_space):
    """⚠ The premise of the whole module, asserted rather than asserted-in-prose: for two
    equivalent centres the SCF's answer to "which atom is this orbital on" is *both*, and no
    threshold on a population can separate them because there is nothing to separate."""
    pops = fragment_populations(h2.spinors_in_ao(), h2.data.s_ao, h2.ao_layout, [0, 1],
                                np.asarray(h2_space.spaces.active))
    np.testing.assert_allclose(pops, 0.5, atol=0.02)


def test_a_per_fragment_character_selection_cannot_be_made_on_equivalent_centres(h2):
    """The other half of the premise: asking for "one orbital on each H" by character is
    refused, because both fragments qualify on the same pair. That refusal is what
    localization exists to answer, so it must keep happening."""
    with pytest.raises(ValueError, match="both claim"):
        active_space_for(h2, character=[(0, "s", 2), (1, "s", 2)], n_active_elec=2)


# --- what localization changes, and what it must not ---------------------------------------

def test_localized_orbitals_sit_on_one_centre_each(h2, h2_space):
    loc = localize_active_space(h2, h2_space, [0, 1], report=False)
    assert loc.weakest > 0.99
    assert loc.site.tolist() == [0, 0, 1, 1]                  # site-blocked order
    assert loc.site_labels == ("1 H", "2 H")
    np.testing.assert_array_equal(np.sort(np.concatenate(
        [loc.site_columns(0), loc.site_columns(1)])), np.asarray(h2_space.spaces.active))


def test_the_ci_energy_is_exactly_invariant(h2, h2_space):
    """⚠ The licence for the whole operation. A CASCI is invariant under a unitary mixing of
    active orbitals, so a localization that moved an energy would not be a localization — it
    would be a different active space wearing the same name. Machine precision, not a
    tolerance: the two calculations are the same one in different coordinates."""
    loc = localize_active_space(h2, h2_space, [0, 1], report=False)
    plain = casci(h2, active=h2_space, n_states=3, report=False)
    local = casci(h2, active=h2_space, n_states=3, coeff=loc.coeff, report=False)
    np.testing.assert_allclose(np.asarray(local.energies), np.asarray(plain.energies),
                               rtol=0.0, atol=1e-12)


def test_the_rotation_is_unitary_and_the_partition_an_exact_cover(h2, h2_space):
    loc = localize_active_space(h2, h2_space, [0, 1], report=False, repair_pairing=False)
    u = loc.rotation
    np.testing.assert_allclose(u.conj().T @ u, np.eye(u.shape[0]), atol=1e-12)
    assert sorted(loc.site.tolist()) == loc.site.tolist()     # blocked, not interleaved
    assert loc.site.size == np.asarray(h2_space.spaces.active).size


def test_kramers_pairs_are_rebuilt_inside_each_site(h2, h2_space):
    """⚠ A localizing rotation is free to mix the members of a pair, like any other
    active-active rotation, and the active-space convention is addressed in pairs. Each site's
    span is time-reversal closed, so the repair is a rotation *inside* a site: it restores the
    pairs and moves no population, and both halves are asserted.

    ⚠ **What is deliberately not asserted: that the raw set comes back unpaired.** The site
    projector's eigenvalues are Kramers degenerate, so ``eigh`` returns an arbitrary basis of
    each pair — on a problem this small it lands on a pair-aligned one perhaps a third of the
    time, run to run. The repair exists because the raw set *may* be unpaired, not because it
    always is, and a test asserting the coin flip would fail for the wrong reason.
    """
    raw = localize_active_space(h2, h2_space, [0, 1], report=False, repair_pairing=False)
    fixed = localize_active_space(h2, h2_space, [0, 1], report=False)
    active = np.asarray(h2_space.spaces.active)

    def partner_deviation(coeff):
        """|barred - T(unbarred)| over the pairs of the active columns, in the AO basis with
        its own metric — the working-basis form is what the reference reports."""
        c = coeff[:, active]
        t = time_reverse(np.ascontiguousarray(c[:, 0::2]))
        return float(np.max(np.abs(c[:, 1::2] - t)))

    # In the AO basis time reversal is still coefficient conjugation (a real scalar basis),
    # so the comparison is meaningful on either representation.
    assert partner_deviation(fixed.coeff) < 1e-10
    np.testing.assert_allclose(np.sort(fixed.populations, axis=None),
                               np.sort(raw.populations, axis=None), atol=1e-8)


# --- the refusals --------------------------------------------------------------------------

def test_a_set_that_does_not_localize_is_refused_with_the_table(h2, h2_space):
    """⚠ The failure this module exists to make loud: a localization that leaves every
    orbital half on each site has produced a site partition in name only, and a
    broken-symmetry guess built on it converges straight back to the symmetric solution. The
    refusal names the knob and prints the populations, because "these are not site orbitals"
    is the answer the user needs."""
    close = spinor_reference(
        Molecule.from_xyz_string("H 0 0 0\nH 0 0 0.74", basis="x2c-SVPall-2c"),
        screening="none")
    space = active_space_for(close, character=([0, 1], "s"), n_active=4, n_active_elec=2)
    localize_active_space(close, space, [0, 1], report=False)      # a bond still localizes
    with pytest.raises(ValueError, match="min_population"):
        localize_active_space(close, space, [0, 1], min_population=0.999, report=False)


def test_counts_must_partition_the_localized_set(h2, h2_space):
    with pytest.raises(ValueError, match="do not divide evenly"):
        localize(h2.spinors_in_ao(), h2.data.s_ao, h2.ao_layout,
                 np.asarray(h2_space.spaces.active)[:3], [0, 1])
    with pytest.raises(ValueError, match="sum to"):
        localize_active_space(h2, h2_space, [0, 1], counts=[1, 1], report=False)


def test_an_odd_site_count_is_refused_when_pairs_are_repaired(h2, h2_space):
    """Half a Kramers pair belongs to nothing, so a 3+1 split of four spinors cannot be
    delivered as a paired set — refused rather than handed back unpaired.

    ``min_population=0`` gets past the population floor on purpose: a 3+1 split of this
    system is *also* physically wrong, and with the floor in force that refusal fires first
    and this guard would never be reached.
    """
    with pytest.raises(ValueError, match="even number of orbitals"):
        localize_active_space(h2, h2_space, [0, 1], counts=[3, 1], min_population=0.0,
                              report=False)


def test_one_site_is_not_a_partition(h2, h2_space):
    with pytest.raises(ValueError, match="at least two sites"):
        localize_active_space(h2, h2_space, [0], report=False)


def test_a_site_with_no_functions_is_refused(h2, h2_space):
    with pytest.raises(ValueError, match="no atom"):
        localize_active_space(h2, h2_space, [0, "Cl"], report=False)


# --- the scalar path, which the broken-symmetry guess runs on -------------------------------

def test_scalar_orbitals_localize_through_the_same_kernel(h2):
    """⚠ Same code on ``(nao, nmo)`` real MOs: the broken-symmetry guess localizes the
    singly-occupied *scalar* orbitals of a high-spin solution, and a second implementation of
    "which centre" for that case is exactly what this module exists to prevent."""
    c = np.asarray(h2.data.mo_coeff)
    assert c.shape[0] == h2.data.nao                       # scalar rows, not spin-blocked
    loc = localize(c, h2.data.s_ao, h2.ao_layout, [0, 1], [0, 1])
    assert loc.weakest > 0.99 and loc.site.tolist() == [0, 1]
    np.testing.assert_allclose(loc.coeff[:, 2:], c[:, 2:], atol=0.0)   # untouched columns


def test_two_far_metal_ions_separate_exactly():
    """The polynuclear case in its cleanest form: two Ti(3+) at 12 A, whose d shells overlap
    not at all. Localization must then be exact — ten orbitals wholly on one metal and ten on
    the other, out of a canonical set that is free to be anything.

    ⚠ Note what is *not* asserted here, and why: at this separation the two d shells are
    numerically degenerate, so the canonical orbitals are an arbitrary mixture rather than
    the clean half-and-half of the H2 fixture — some come out delocalized and some already
    localized, run to run. That is precisely why a partition cannot be read off the canonical
    set even when the physics is trivial.
    """
    mol = Molecule([("Ti", (0.0, 0.0, 0.0)), ("Ti", (12.0, 0.0, 0.0))],
                   basis="x2c-SVPall-2c", charge=6, spin=2)
    ref = spinor_reference(mol, screening="none")
    space = active_space_for(ref, character=([0, 1], "d"), n_active=20, n_active_elec=2)
    loc = localize_active_space(ref, space, [0, 1], report=False)
    assert loc.weakest > 0.999
    assert loc.site.tolist() == [0] * 10 + [1] * 10
