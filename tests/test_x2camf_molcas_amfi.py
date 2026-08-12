"""The method-decomposition table, asserted.

``tests/reference/x2camf_molcas_amfi.json`` records, for the same ion in the same basis, the
splitting from one-electron X2C, from Kuiva's X2CAMF, from OpenMolcas's DKH2+AMFI, from a
four-component solve, and from experiment. These tests assert the *relationships* between
those columns, because that is what the table is for — an absolute band on any one of them
would be a statement about a small basis, not about a method.

⚠ **The one thing not asserted here is agreement with OpenMolcas**, and that is deliberate.
The two differ by 17-19% and the table cannot attribute it: DKH2-vs-X2C, Breit-Pauli-AMFI-vs-
X2CAMF and the orbitals all change at once. What *is* asserted is the thing that can be
attributed — that Kuiva reproduces four-component Dirac-Coulomb in the same basis, which is
the theory X2C approximates and which contains no picture change to get wrong.

The reference is generated once (``tests/generate/x2camf_molcas_amfi.py``) and needs
OpenMolcas; these tests read the committed file and need nothing.
"""
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REFERENCE = REPO / "tests/reference/x2camf_molcas_amfi.json"

#: Kuiva's X2CAMF against its own four-component solve **in the same (contracted) basis**.
#: Observed -0.42% (Ne) and -1.29% (Ar). ⚠ The comparison must be contracted-vs-contracted:
#: against the *primitive* four-component number Ne would read +3.0%, which is the mixed-basis trap, not a method error.
FOUR_COMPONENT_TOL_RELATIVE = 0.03

#: How far below Kuiva the DKH2+AMFI column sits. A **band**, not a bound: it is a real
#: method difference (observed 24.2% for Ne, 16.8% for Ar), so both its disappearance and its
#: growth would be news. This is the number the table exists to record.
AMFI_GAP_BAND = (0.05, 0.45)


def records():
    if not REFERENCE.is_file():
        pytest.skip("{} not generated".format(REFERENCE.name))
    return json.loads(REFERENCE.read_text())["records"]


def ok_records():
    return {k: v for k, v in records().items() if v.get("status") == "ok"}


IONS = ["Ne", "Ar"]


@pytest.mark.parametrize("symbol", IONS)
def test_x2camf_reproduces_four_component_theory_in_the_same_basis(symbol):
    """The assertion that can actually be attributed to a method.

    Everything else in this table changes three things at once; this changes one — whether the
    picture change is performed — and a four-component calculation does not perform it at all.
    """
    r = ok_records().get(symbol) or pytest.skip("no record for {}".format(symbol))
    s = r["splitting_cm"]
    rel = abs(s["kuiva_coulomb"] - s["four_component_contracted"]) / s["four_component_contracted"]
    assert rel < FOUR_COMPONENT_TOL_RELATIVE, \
        "{}: X2CAMF {:.2f} vs four-component {:.2f} cm^-1 ({:.2%})".format(
            symbol, s["kuiva_coulomb"], s["four_component_contracted"], rel)


@pytest.mark.parametrize("symbol", IONS)
def test_the_correction_moves_the_splitting_toward_four_component_theory(symbol):
    """The one-electron operator overshoots and the correction removes most of the overshoot.

    Asserted as *"the corrected number is closer"* rather than as a band, so it stays a
    statement about the correction and cannot be satisfied by a basis coincidence.
    """
    s = (ok_records().get(symbol) or pytest.skip("no record"))["splitting_cm"]
    ref = s["four_component_contracted"]
    assert abs(s["kuiva_coulomb"] - ref) < 0.1 * abs(s["one_electron_x2c"] - ref)


@pytest.mark.parametrize("symbol", IONS)
def test_gaunt_lowers_the_splitting(symbol):
    """Spin-other-orbit is a second screening channel, so adding Gaunt must reduce it.

    ⚠ And the *relative* effect shrinks with Z : 8.0% at Ne, 3.6% at Ar
    here. A sign flip would be the convention error that is invisible to every
    norm-based check.
    """
    s = (ok_records().get(symbol) or pytest.skip("no record"))["splitting_cm"]
    assert s["kuiva_gaunt"] < s["kuiva_coulomb"]


@pytest.mark.parametrize("symbol", IONS)
def test_the_dkh2_amfi_column_differs_by_a_recorded_amount(symbol):
    """⚠ The deliverable itself: the size of the disagreement with the mainstream SMM route.

    Recorded rather than explained, because this table changes three variables at once and
    cannot attribute it. A band on both sides, so that a future change which *removed* the
    difference would fail here and have to be understood rather than silently accepted.
    """
    s = (ok_records().get(symbol) or pytest.skip("no record"))["splitting_cm"]
    gap = (s["kuiva_coulomb"] - s["molcas_dkh2_amfi"]) / s["kuiva_coulomb"]
    lo, hi = AMFI_GAP_BAND
    assert lo < gap < hi, "{}: DKH2+AMFI is {:.1%} below X2CAMF ({:.2f} vs {:.2f} cm^-1)"\
        .format(symbol, gap, s["molcas_dkh2_amfi"], s["kuiva_coulomb"])


@pytest.mark.parametrize("symbol", IONS)
def test_the_spin_orbit_manifold_split_four_plus_two(symbol):
    """A ²P ion must split 4 + 2. Anything else means the OpenMolcas run was not on the state
    the input said, and its splitting would then be a plausible number for the wrong thing."""
    m = (ok_records().get(symbol) or pytest.skip("no record"))["molcas"]
    assert m["degeneracies"] == [4, 2]


def test_the_reference_says_what_it_cannot_separate():
    """The caption is part of the deliverable: this table must never be
    read as a controlled single-variable comparison, so the file has to say so itself."""
    document = json.loads(REFERENCE.read_text()) if REFERENCE.is_file() else pytest.skip("-")
    confound = document["confound"]
    for phrase in ("DKH2", "AMFI", "cannot separate"):
        assert phrase in confound


def test_both_four_component_columns_are_recorded():
    """⚠ Contracted **and** primitive, because quoting a contracted two-component number
    against a primitive four-component one is the mixed-basis comparison trap. Their difference
    is the basis-set error and is 3.4% for Ne here — larger than the method residual it would
    otherwise be mistaken for."""
    for symbol, r in ok_records().items():
        s = r["splitting_cm"]
        assert "four_component_contracted" in s and "four_component_primitive" in s, symbol


def test_the_mean_field_reference_is_the_neutral_atom_on_the_kuiva_side():
    """AMFI's atomic mean field is not the ion's either, and the measured sensitivity to that
    choice is 0.21% (3d) to 13 ppm (4f) — so it is recorded to rule it out as an explanation,
    not because it is one."""
    from kuiva.amf.configuration import AtomicConfiguration

    for symbol, r in ok_records().items():
        assert r["mean_field_reference"] == AtomicConfiguration.ground(symbol).canonical
        # ⚠ The canonical form counts electrons per **angular-momentum channel**, not per
        # shell, so Ar(+) is "s6 p11" and not "... p5" — the ion is one p electron short of
        # the neutral atom, which is what is asserted.
        assert r["state"] == AtomicConfiguration.for_oxidation_state(symbol, 1).canonical
        neutral = AtomicConfiguration.ground(symbol)
        assert (AtomicConfiguration.for_oxidation_state(symbol, 1).n_electrons
                == neutral.n_electrons - 1)
