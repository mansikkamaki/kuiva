"""The custom-basis escape hatch, and the discipline it is required to keep.

The registry exists so that a calculation cannot silently run with a basis that does not suit
it. A user-supplied set goes around the *names* and must not go around the *checks*, so the
tests here are in two halves: that a custom set is genuinely the same set (a registered family
pushed through this path must reproduce itself to machine precision, integrals and energy
alike), and that everything the registry would have refused is still refused.

⚠ The one property that cannot be measured from shells is the **relativistic treatment**, and
it is the one whose absence produces plausible wrong numbers rather than an error. It is
therefore required, and a test holds that requirement in place.
"""
from __future__ import annotations

import numpy as np
import pytest

import kuiva
from kuiva.basis.custom import CustomBasis, is_custom, is_custom_name, stash, unstash
from kuiva.basis.registry import Contraction
from kuiva.interface.pyscf_bridge import build_mole

BASIS = "x2c-SVPall-2c"

pytestmark = pytest.mark.stage_under_test("amf_atomic")


def shells_of(element, family=BASIS):
    """The parsed shells a registered family gives an element, as the front end builds them."""
    mol = build_mole(kuiva.Molecule(atoms=[(element, (0., 0., 0.))], basis=family,
                                    spin=_spin(element)))
    return mol._basis[mol.atom_symbol(0)]


def _spin(element):
    from pyscf import gto
    return int(gto.charge(element)) % 2


# --- It is the same basis ------------------------------------------------------------------

def test_a_registered_family_round_trips_through_the_custom_path():
    """⚠ The load-bearing test: same functions in, same functions out. Not "close" — the
    integrals must be **bitwise** identical, because the custom path is meant to be a way of
    naming a basis and not a way of approximating one."""
    reg_mol = build_mole(kuiva.Molecule(atoms=[("Ne", (0., 0., 0.))], basis=BASIS))
    custom = CustomBasis(data={"Ne": shells_of("Ne")}, relativistic_treatment="x2c-2c",
                         name="svpall-copy")
    cus_mol = build_mole(kuiva.Molecule(atoms=[("Ne", (0., 0., 0.))], basis=custom))
    assert cus_mol.nao == reg_mol.nao
    assert np.array_equal(cus_mol.intor("int1e_ovlp"), reg_mol.intor("int1e_ovlp"))
    assert np.array_equal(cus_mol.intor("int1e_nuc"), reg_mol.intor("int1e_nuc"))


def test_the_scf_energy_matches_the_registered_family():
    custom = CustomBasis(data={"Ne": shells_of("Ne")}, relativistic_treatment="x2c-2c",
                         name="svpall-copy")
    a = kuiva.ScalarSCF(kuiva.Molecule(atoms=[("Ne", (0., 0., 0.))], basis=BASIS),
                        memory_gb=4.0, screening="none").run()
    b = kuiva.ScalarSCF(kuiva.Molecule(atoms=[("Ne", (0., 0., 0.))], basis=custom),
                        memory_gb=4.0, screening="none").run()
    assert b.energy == pytest.approx(a.energy, abs=1e-11)


def test_the_atomic_mean_field_works_over_a_custom_basis():
    """⚠ Its cache keys on the **content** of the parsed shells rather than on a name, which
    is exactly what makes this work: a custom set and the registered family it copies produce
    the identical correction, and the identity is bitwise."""
    custom = CustomBasis(data={"Ne": shells_of("Ne")}, relativistic_treatment="x2c-2c",
                         name="svpall-copy")
    a = kuiva.ScalarSCF(kuiva.Molecule(atoms=[("Ne", (0., 0., 0.))], basis=BASIS),
                        memory_gb=4.0).run()
    b = kuiva.ScalarSCF(kuiva.Molecule(atoms=[("Ne", (0., 0., 0.))], basis=custom),
                        memory_gb=4.0).run()
    assert np.max(np.abs(a.data.soc.w - b.data.soc.w)) == 0.0


def test_a_custom_basis_survives_the_scf_checkpoint():
    """⚠ The trap this feature fell into first, and it is not hypothetical: the integral
    library's SCF checkpoint JSON-serializes the molecule's ``__dict__``, so a basis object
    parked there fails the run from inside a JSON encoder with a message about nothing.
    Running an SCF at all is the test; what is stashed is plain data."""
    custom = CustomBasis(data={"Ne": shells_of("Ne")}, relativistic_treatment="x2c-2c",
                         name="svpall-copy")
    mol = build_mole(kuiva.Molecule(atoms=[("Ne", (0., 0., 0.))], basis=custom))
    import json

    json.dumps(mol.__dict__["_kuiva_atom_families"])       # the actual invariant
    json.dumps(mol.__dict__["_kuiva_atom_basis"])
    assert is_custom_name(mol.__dict__["_kuiva_atom_basis"]["Ne"])
    assert is_custom(unstash(stash(custom)))
    assert unstash(stash(custom)).digest() == custom.digest()


# --- The checks it must not go around ------------------------------------------------------

def test_the_relativistic_treatment_is_required():
    """It cannot be measured from a list of exponents, and it is the one property whose
    absence stays invisible until it reaches a heavy element."""
    with pytest.raises(ValueError, match="relativistic_treatment"):
        CustomBasis(data={"Ne": shells_of("Ne")})
    with pytest.raises(ValueError, match="unknown relativistic_treatment"):
        CustomBasis(data={"Ne": shells_of("Ne")}, relativistic_treatment="relativistic-ish")


def test_mixing_a_non_relativistic_custom_set_into_an_x2c_calculation_is_refused():
    """The silent-error trap the registry exists to prevent, reached through the escape
    hatch: the check is the same check because a custom set answers the same question."""
    custom = CustomBasis(data={"Ne": shells_of("Ne")}, relativistic_treatment="nonrel",
                         name="pretend-nonrel")
    with pytest.raises(ValueError, match="consistency"):
        kuiva.Molecule(atoms=[("Ne", (0., 0., 0.)), ("Ar", (0., 0., 4.))],
                       basis={"Ne": custom, "Ar": BASIS})


def test_coverage_is_measured_from_the_data():
    custom = CustomBasis(data={"Ne": shells_of("Ne")}, relativistic_treatment="x2c-2c")
    assert custom.covers("Ne") and not custom.covers("Ar")
    with pytest.raises(ValueError, match="consistency|does not cover"):
        kuiva.Molecule(atoms=[("Ar", (0., 0., 0.))], basis=custom)


def test_the_contraction_type_is_measured_not_declared():
    """The existing rule, applied to data that has no family name to be believed about.
    Karlsruhe is segmented and Dyall is uncontracted, and neither says so anywhere in the
    numbers a user hands over."""
    seg = CustomBasis(data={"Ne": shells_of("Ne", "x2c-SVPall-2c")},
                      relativistic_treatment="x2c-2c")
    unc = CustomBasis(data={"Ne": shells_of("Ne", "dyallv2z")},
                      relativistic_treatment="x2c-2c")
    assert seg.contraction_of("Ne") is Contraction.SEGMENTED
    assert unc.contraction_of("Ne") is Contraction.UNCONTRACTED
    assert "segmented" in seg.label("Ne") and "cholesky" in seg.label("Ne")


def test_an_nwchem_string_is_accepted():
    """The form published sets are actually distributed in."""
    text = """
BASIS "ao basis" PRINT
He    S
      6.36242139              0.15432897
      1.15892300              0.53532814
      0.31364979              0.44463454
END
"""
    custom = CustomBasis(data=text, relativistic_treatment="nonrel", name="he-sto3g")
    assert custom.covers("He") and not custom.covers("Ne")
    mol = build_mole(kuiva.Molecule(atoms=[("He", (0., 0., 0.))], basis=custom))
    assert mol.nao == 1


def test_a_custom_set_mixes_with_registered_families_per_atom():
    """Through the same per-atom addressing everything else uses — which is the point of
    putting it on that axis rather than making it a mode."""
    custom = CustomBasis(data={"He": shells_of("He")}, relativistic_treatment="x2c-2c",
                         name="he-copy")
    run = kuiva.ScalarSCF(kuiva.Molecule(
        atoms=[("He", (0., 0., 0.)), ("Ne", (0., 0., 4.))],
        basis={"He": custom, "Ne": BASIS}), memory_gb=4.0, screening="none").run()
    meta = run.data.basis_meta
    assert any(v.startswith("custom:") for v in meta.values())
    assert any(v.startswith("x2c-SVPall-2c") for v in meta.values())
    assert run.converged
