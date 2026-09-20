"""The nuclear-moment table and the atomic-unit constants beside it.

⚠ **Weighed by what each check can fail on**, because a wrong ``g`` factor produces a
hyperfine coupling that is Hermitian, of a plausible magnitude, wrong, and written into a file
that outlives the session:

* **Parity.** ``2I`` and the mass number have the same parity for every nucleus — an odd-``A``
  nuclide has half-integer spin, an even-``A`` one integer spin. That is a theorem about
  nucleon counting, it holds for every entry in the table, and it fails on a typo in either
  column.
* **An independent compilation.** Every ``(2I, g)`` pair is compared against PySCF's shipped
  table, which is derived from the EasySpin isotope tables rather than from the sources cited
  in :mod:`kuiva.util.nuclei`. ⚠ The two share ultimate provenance (Stone's compilation), so
  this catches **transcription errors and nothing more** — which is exactly what it is for,
  and it is why the parity check and the physical spot values below are separate checks.
* **A physical spot value.** The ``1``H hyperfine coupling constant follows from ``g(1H)``,
  :data:`~kuiva.util.units.NUCLEAR_MAGNETON` and the hydrogenic ``|psi(0)|^2`` alone, and
  comes out at the measured 1420 MHz. It is the one check here that tests the *constants*
  rather than the table.
* **Refusals**, all three of which exist because the silent alternative is a plausible wrong
  number: an unknown isotope, a label naming the wrong element, and ``I = 0``.
"""
import numpy as np
import pytest

from kuiva.util import nuclei
from kuiva.util.nuclei import (ISOTOPES, NuclearMoment, covered_elements, default_isotope,
                               isotopes, parse_isotope_label, resolve_isotope)
from kuiva.util.units import (BARN_TO_BOHR2, G_ELECTRON, HARTREE_TO_MHZ, NUCLEAR_MAGNETON,
                              PROTON_ELECTRON_MASS_RATIO)


# --- the table's internal consistency -------------------------------------------------------

def test_every_entry_has_the_parity_its_mass_number_demands():
    """``2I`` and ``A`` share parity — a theorem about nucleon counting, not a convention."""
    bad = [(sym, m.label, m.twice_spin) for sym, ms in ISOTOPES.items() for m in ms
           if (m.mass_number % 2) != (m.twice_spin % 2)]
    assert bad == []


def test_entries_are_ordered_and_unique_per_element():
    for sym, ms in ISOTOPES.items():
        masses = [m.mass_number for m in ms]
        assert masses == sorted(masses), sym
        assert len(set(masses)) == len(masses), sym
        assert all(m.symbol == sym for m in ms)


def test_a_zero_spin_entry_carries_no_moment_and_no_quadrupole():
    """A listed ``I = 0`` nuclide exists only so that naming it is refused informatively."""
    for sym, ms in ISOTOPES.items():
        for m in ms:
            if m.twice_spin == 0:
                assert m.g == 0.0 and m.quadrupole_barn is None, (sym, m.label)
                assert not m.magnetic


def test_a_quadrupole_moment_needs_spin_one_or_more():
    """``Q`` vanishes identically for ``I < 1``; a value there would be a misfiled entry."""
    for sym, ms in ISOTOPES.items():
        for m in ms:
            if m.twice_spin < 2:
                assert m.quadrupole_barn is None, (sym, m.label)
            assert m.has_quadrupole == (m.twice_spin >= 2
                                        and m.quadrupole_barn is not None)


def test_abundances_are_percentages_and_sum_below_a_hundred():
    """The table is curated, so the listed isotopes of an element need not be all of them —
    but they can never sum to more than 100%."""
    for sym, ms in ISOTOPES.items():
        total = sum(m.abundance for m in ms)
        assert 0.0 <= total <= 100.0001, (sym, total)


def test_the_program_targets_are_covered():
    """⚠ A coverage claim, asserted rather than described: the elements this program targets
    -- 3d/4d/5d transition metals, lanthanides, the early actinides -- plus the nuclei that
    appear as their ligands. A table that quietly lost the lanthanides
    would still pass every check above."""
    lanthanides = "La Ce Pr Nd Sm Eu Gd Tb Dy Ho Er Tm Yb Lu".split()
    three_d = "Sc Ti V Cr Mn Fe Co Ni Cu Zn".split()
    five_d = "Hf Ta W Re Os Ir Pt Au Hg".split()
    ligands = "H B C N O F Al Si P S Cl As Se Br I".split()
    actinides = "Th U Np Pu Am".split()
    missing = [e for e in lanthanides + three_d + five_d + ligands + actinides
               if e not in ISOTOPES]
    assert missing == []
    assert covered_elements() == tuple(sorted(ISOTOPES))


# --- against an independent compilation -----------------------------------------------------

def test_spins_and_g_factors_match_an_independent_compilation():
    """Cross-check against PySCF's shipped ``ISOTOPE_GYRO``, which is EasySpin-derived.

    ⚠ **This is a transcription check, not evidence that the physics is right.** Both tables
    trace back to Stone's compilation, so agreement says the digits were copied correctly and
    nothing else; it is paired with the parity check above and the 1420 MHz spot value below,
    which can fail on things this cannot.

    PySCF lists exactly one isotope per element — the NMR-active one — so the comparison runs
    over the isotopes both tables name, and a Kuiva entry PySCF does not carry is simply not
    compared.
    """
    from pyscf.data.elements import ELEMENTS
    from pyscf.data.nucprop import ISOTOPE_GYRO

    compared = 0
    for sym, ms in ISOTOPES.items():
        z = ELEMENTS.index(sym)
        for mass, spin, g in ISOTOPE_GYRO[z]:
            ours = [m for m in ms if m.mass_number == mass]
            if not ours:
                continue
            m = ours[0]
            # ⚠ Two deliberate divergences, both of which are PySCF placeholders rather than
            # data: an entry whose g is exactly zero at a nonzero spin says "not tabulated".
            if g == 0.0 and spin != 0:
                continue
            compared += 1
            assert m.twice_spin == int(round(2 * spin)), (sym, mass)
            assert m.g == pytest.approx(g, rel=2e-4, abs=1e-4), (sym, mass, m.g, g)
    # A check that silently compared nothing would pass for the wrong reason.
    assert compared >= 60


def test_quadrupole_magnitudes_match_an_independent_compilation():
    """Same idea for ``Q``, on **magnitudes only**, and that restriction is the finding.

    ⚠ PySCF's quadrupole table carries several moments whose published sign is negative as
    positive numbers (``17``O, ``33``S, ``35``Cl, ``127``I, ``209``Bi are all negative in
    Pyykkö 2018). Comparing signed values would fail on entries this table has deliberately
    right, so the sign is checked against the source in review and the magnitude here.

    ⚠ The band is 1%, not the 2e-4 the ``g`` factors get, and that difference is itself a
    fact: the two tables are different *revisions* of the quadrupole compilation and genuinely
    differ by up to 0.7% (``59``Co, 0.420 against 0.423 b). A transcription error is a wrong
    digit — tens of per cent — so the check still does its job.
    """
    from pyscf.data.elements import ELEMENTS
    from pyscf.data.nucprop import ISOTOPE_QUAD_MOMENT

    compared = 0
    for sym, ms in ISOTOPES.items():
        z = ELEMENTS.index(sym)
        if z >= len(ISOTOPE_QUAD_MOMENT):
            continue
        mass, spin, q = ISOTOPE_QUAD_MOMENT[z]
        ours = [m for m in ms if m.mass_number == mass and m.quadrupole_barn is not None]
        if not ours or q == 0.0:
            continue
        compared += 1
        assert abs(ours[0].quadrupole_barn) == pytest.approx(abs(q), rel=1e-2), (sym, mass)
    assert compared >= 15


# --- the constants, against a measured number -----------------------------------------------

def test_the_hydrogen_hyperfine_constant_comes_out_of_the_constants_alone():
    """``a(1H) = (8 pi/3) alpha^2 (g_e/2) g_N mu_N |psi(0)|^2`` = the measured 1420 MHz.

    ⚠ This is the check that tests the **constants** rather than the table: it uses
    :data:`NUCLEAR_MAGNETON` (whose factor of one half is the easiest thing in this project to
    drop), :data:`HARTREE_TO_MHZ`, :data:`G_ELECTRON` and ``g(1H)``, and nothing else. The
    analytic hydrogenic ``|psi(0)|^2 = 1/pi`` makes it independent of every integral.
    """
    alpha2 = 1.0 / 137.035999084 ** 2
    g_n = resolve_isotope("H", 1).g
    a_hartree = (8.0 * np.pi / 3.0) * alpha2 * (G_ELECTRON / 2.0) * g_n * NUCLEAR_MAGNETON \
        / np.pi
    assert a_hartree * HARTREE_TO_MHZ == pytest.approx(1420.4, rel=3e-3)


def test_the_nuclear_magneton_is_the_bohr_magneton_times_the_mass_ratio():
    assert NUCLEAR_MAGNETON == pytest.approx(0.5 / PROTON_ELECTRON_MASS_RATIO, rel=0, abs=0)
    assert NUCLEAR_MAGNETON == pytest.approx(2.723085e-4, rel=1e-6)


def test_the_barn_is_a_barn():
    """``1 b = 1e-28 m^2``; in bohr^2 that is 3.5711e-8, and getting it wrong by ``a_0``
    rather than ``a_0^2`` is a factor of 1.9e10 that looks like a unit bug in the caller."""
    assert BARN_TO_BOHR2 == pytest.approx(3.571065e-8, rel=1e-6)


def test_g_electron_is_one_literal_shared_with_the_property_layer():
    from kuiva.props.multiplet import G_ELECTRON as from_props
    assert from_props is G_ELECTRON


# --- resolution -----------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("159Tb", ("Tb", 159)), ("Tb159", ("Tb", 159)), ("Tb-159", ("Tb", 159)),
    ("159-Tb", ("Tb", 159)), ("159", (None, 159)), (" 163dy ", ("Dy", 163)),
])
def test_isotope_labels_read_in_either_order(text, expected):
    assert parse_isotope_label(text) == expected


@pytest.mark.parametrize("text", ["Tb", "", "1590Tb", "Tb1.5", "159 Tb Dy"])
def test_an_unreadable_isotope_label_is_refused(text):
    with pytest.raises(ValueError, match="isotope"):
        parse_isotope_label(text)


@pytest.mark.parametrize("spec", [True, 159, "159", "159Tb", "Tb-159"])
def test_every_spelling_of_the_same_isotope_resolves_to_one_object(spec):
    m = resolve_isotope("Tb", spec)
    assert m.label == "159Tb" and m.twice_spin == 3
    assert m.multiplicity == 4
    assert m.moment_nuclear_magnetons == pytest.approx(1.343 * 1.5)


def test_the_default_isotope_is_the_most_abundant_one_that_has_a_moment():
    """⚠ Not the most abundant nuclide: carbon's is ``12``C, which has no moment at all."""
    assert default_isotope("C").label == "13C"
    assert default_isotope("O").label == "17O"
    assert default_isotope("Fe").label == "57Fe"
    assert default_isotope("Cu").label == "63Cu"          # 69.15% against 65Cu's 30.85%
    assert default_isotope("Dy").label == "163Dy"          # 24.90% against 161Dy's 18.91%
    assert default_isotope("Tl").label == "205Tl"


def test_a_synthetic_default_isotope_warns(kuiva_caplog):
    """Th has no naturally occurring isotope with a moment; the default is real and is a
    surprise, so it announces itself."""
    assert default_isotope("Th").label == "229Th"
    assert "does not occur naturally" in kuiva_caplog.text


def test_an_element_with_no_magnetic_isotope_is_refused_by_name():
    """Cerium's stable nuclides all have ``I = 0``. ⚠ The refusal has to say *that*, not
    "unknown element" — a lanthanide chemist will try it."""
    with pytest.raises(ValueError, match="no tabulated isotope with a nuclear moment"):
        default_isotope("Ce")


def test_a_zero_spin_isotope_is_refused_by_name():
    with pytest.raises(ValueError, match="has I = 0"):
        resolve_isotope("O", 16)
    with pytest.raises(ValueError, match="has I = 0"):
        resolve_isotope("Pb", "208Pb")


def test_a_label_naming_another_element_is_refused_not_reinterpreted():
    """``{"Tb1": "163Dy"}`` is a mistake. Reading it as "mass 163 of terbium" would either
    refuse for the wrong reason or, worse, find an entry."""
    with pytest.raises(ValueError, match="refused rather than reinterpreted"):
        resolve_isotope("Tb", "163Dy")


def test_an_untabulated_isotope_is_refused_with_what_is_available():
    with pytest.raises(ValueError, match="no nuclear moment is tabulated for mass number 158"):
        resolve_isotope("Tb", 158)


def test_an_untabulated_element_names_the_escape_hatch():
    with pytest.raises(ValueError, match="NuclearMoment"):
        isotopes("Rn")


@pytest.mark.parametrize("spec", [None, False, 1.5, ("Tb", 159)])
def test_a_nonsense_request_is_refused(spec):
    with pytest.raises((ValueError, TypeError)):
        resolve_isotope("Tb", spec)


# --- the user-supplied moment ---------------------------------------------------------------

def test_a_user_supplied_moment_is_used_as_given_and_says_so():
    m = NuclearMoment(spin=3.5, g=0.5, quadrupole_barn=-1.25)
    assert resolve_isotope("Cm", m) is m                  # an element the table lacks
    assert m.twice_spin == 7 and m.multiplicity == 8
    assert "custom" in m.label and m.source == "user-supplied"
    assert m.as_dict()["element"] is None


def test_a_spin_that_is_not_a_multiple_of_one_half_is_refused():
    for bad in (0.3, -0.5, 1.2):
        with pytest.raises(ValueError, match="multiple of one half"):
            NuclearMoment(spin=bad, g=1.0)


def test_a_tabulated_moment_records_where_it_came_from():
    m = resolve_isotope("Tb", True)
    assert "Stone" in m.source and "Pyykko" in m.source
    d = m.as_dict()
    assert d["twice_spin"] == 3 and d["mass_number"] == 159 and d["element"] == "Tb"
    assert nuclei.MOMENT_SOURCE in m.source
