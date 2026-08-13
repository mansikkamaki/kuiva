"""The Slater-Condon driver, its result object, and the file it writes.

Stages A-E and the spin-orbit fit are tested against physics elsewhere in this suite. What is
left, and what this file is about, is the part that outlives the session: a **stored product**
that an external program — or the same user in a year — reads without the code that wrote it.
Three things have to hold for that to be safe, and each has a test here:

1. **It round-trips.** A format nobody has ever read back is a format with an undetected
   ambiguity in it, so the reader is part of the module and is exercised against the writer at
   full precision.
2. ⚠ **It says which Hamiltonian produced it, always.** A spin-orbit constant that does not
   say whether the two-electron screening is in it is not interpretable — the difference is
   5-30% — and the case that matters is the one where there is nothing to say: a reference run
   without spin-orbit ingestion writes a record stating *that*, never an empty section.
3. **It refuses what it cannot read.** The format version exists so a consumer can refuse
   rather than misinterpret, which is only true if refusing is what it actually does.

Plus the driver's own defaults, which are decisions rather than conveniences: which shells are
taken when none are named, and what ``zeta=False`` does to the screening.

Cost: two light atoms in a split-valence basis, one of them with a four-component atomic
solve, about three seconds.
"""
import json

import numpy as np
import pytest

from kuiva.extras.shells import ShellConfiguration
from kuiva.extras.slater_condon import (CONVENTION, FORMAT_VERSION, SlaterCondonResult,
                                        read_parameters, slater_condon_parameters)

BASIS = "x2c-SVPall-2c"
MEMORY_GB = 8.0


@pytest.fixture(scope="module")
def carbon():
    """C 1s^2 2s^2 2p^2 without spin-orbit ingestion — the cheap path, and the common one."""
    return slater_condon_parameters("C", "1s2 2s2 2p2", basis=BASIS, shells=("2s", "2p"),
                                    zeta=False, memory_gb=MEMORY_GB, conv_tol=1e-11,
                                    report=False)


# --- The result ---------------------------------------------------------------------------

def test_the_driver_returns_everything_one_run_produced(carbon):
    """The container, and the two diagnostics that decide whether it means anything.

    ⚠ They are separate numbers because neither implies the other: the anisotropy says whether
    the *solution* was spherical and the residual says whether the *extraction* was consistent
    with it, and a symmetry-broken solution gives a residual at roundoff.
    """
    assert carbon.element == "C"
    assert carbon.charge == 0                        # derived from the electron count
    assert carbon.configuration.canonical == "1s2 2s2 2p2"
    assert carbon.labels == ("2s", "2p")
    assert carbon.data.converged
    assert carbon.anisotropy < 1e-8
    assert carbon.max_relative_residual < 1e-10
    assert len(carbon.parameters) == 5               # F0 x3, F2(2p,2p), G1(2s,2p)
    assert len(carbon.spin_orbit) == 0               # no operator was ingested
    assert "C" in repr(carbon)


def test_the_two_families_of_number_cannot_collide_in_one_record(carbon):
    """``as_dict`` merges parameters and constants, so the constants are keyed ``zeta(2p)``.
    Two families of atomic parameter in one flat record is exactly where a silent overwrite
    would live."""
    values = carbon.as_dict()
    assert set(values) == {"F0(2s,2s)", "F0(2s,2p)", "F0(2p,2p)", "F2(2p,2p)", "G1(2s,2p)"}
    assert all(not key.startswith("zeta") for key in values)


def test_the_default_shells_are_the_open_ones():
    """The default is what a parameter set is normally quoted over: the **open** shells. A
    closed-shell configuration has none, and the fallback is every occupied shell rather than
    an empty extraction that would refuse two frames later."""
    open_shell = slater_condon_parameters("C", "1s2 2s2 2p2", basis=BASIS, zeta=False,
                                          memory_gb=MEMORY_GB, report=False)
    closed = slater_condon_parameters("Be", "1s2 2s2", basis=BASIS, zeta=False,
                                      memory_gb=MEMORY_GB, report=False)

    assert open_shell.labels == ("2p",)
    assert closed.labels == ("1s", "2s")


def test_asking_for_no_zeta_switches_the_screening_off():
    """⚠ Spin-orbit ingestion changes no scalar quantity, so with no constants wanted it is
    pure cost — and the cost is a four-component atomic solve per element, tens of minutes for
    a lanthanide. The default therefore follows ``zeta``, and this is the test that says so.
    """
    result = slater_condon_parameters("C", "1s2 2s2 2p2", basis=BASIS, zeta=False,
                                      memory_gb=MEMORY_GB, report=False)
    assert result.data.soc is None
    assert result.provenance["screening"]["method"] == "none"
    assert "with_soc=False" in json.dumps(result.provenance)


def test_the_result_can_be_built_from_a_solution_that_already_exists(carbon):
    """The half of the driver that is not the SCF, for a caller analysing one solution several
    ways. It must give the same numbers as the driver did."""
    config = ShellConfiguration.parse("1s2 2s2 2p2")
    rebuilt = SlaterCondonResult.from_solution(carbon.data, config, "C",
                                               shells=("2s", "2p"), spin_orbit=True)

    assert rebuilt.as_dict() == carbon.as_dict()
    assert len(rebuilt.spin_orbit) == 0              # asked for, but the operator is not there


def test_the_report_goes_through_the_output_grammar(carbon, kuiva_caplog):
    import logging

    with kuiva_caplog.at_level(logging.INFO):
        carbon.report()
    text = "\n".join(record.getMessage() for record in kuiva_caplog.records)

    assert "Slater-Condon parameters" in text
    assert "atomic shells, average of configuration" in text
    assert "F2(2p,2p)" in text
    # ⚠ Matrices and prose never reach INFO; a table and entry lines are all this may print.
    assert "shell anisotropy" in text


# --- The file -----------------------------------------------------------------------------

def test_the_file_round_trips_at_full_precision(carbon, tmp_path):
    """Written and read back through the module's own parser, value by value.

    The file carries **more precision than the log** on purpose: a radial parameter is
    refitted and recombined by whoever reads it, so hartree is written at twelve digits and
    wavenumbers at four decimals, and both must survive the trip.
    """
    path = carbon.write(tmp_path / "carbon.scp", title="C 2p^2, round-trip test")
    back = read_parameters(path)

    assert back["header"]["format"] == "KUIVA_SLATER_CONDON"
    assert int(back["header"]["format_version"]) == FORMAT_VERSION
    assert back["header"]["element"] == "C"
    assert back["header"]["configuration_canonical"] == "1s2 2s2 2p2"
    assert back["header"]["scf_converged"] == "yes"
    assert float(back["header"]["scf_energy_hartree"]) == pytest.approx(carbon.data.e_scf,
                                                                       rel=1e-12)
    assert set(back["parameters"]) == set(carbon.as_dict())
    for parameter in carbon.parameters:
        stored = back["parameters"][parameter.label]
        assert stored["value"] == pytest.approx(parameter.value, rel=1e-12)
        assert stored["value_cm"] == pytest.approx(parameter.value_cm, abs=5e-5)
        assert stored["shells"] == parameter.shells
        assert stored["k"] == parameter.k
        assert stored["n_equations"] == parameter.n_equations
    assert [s["shell"] for s in back["shells"]] == list(carbon.labels)
    assert back["shells"][1]["electrons"] == pytest.approx(2.0, abs=1e-6)
    assert back["zeta"] == {}


def test_the_file_states_its_conventions_and_its_provenance(carbon, tmp_path):
    """⚠ **Two things a stored parameter file may never leave out**, both because it will be
    read without the code that wrote it.

    The ``R^k(ab;cd)`` **ordering** — different authors order it differently, and a file of
    numbers without the definition is ambiguous rather than merely terse — and the
    **provenance record**, which says whether the two-electron screening is in the constants.
    The provenance is written even when it says ``none``: "no record" and "the record says
    none" are different statements and only one of them is true here.
    """
    text = carbon.write(tmp_path / "carbon.scp").read_text()

    assert CONVENTION.splitlines()[0] in text
    assert "Condon-Shortley ordering" in text
    assert "bare Coulomb integrals over the X2C radial functions" in text
    assert "[PROVENANCE]" in text
    provenance = read_parameters(tmp_path / "carbon.scp")["provenance"]
    assert provenance["screening"]["method"] == "none"
    assert "frozen average-of-configuration" in provenance["construction"]


def test_the_reader_refuses_a_format_version_it_does_not_know(carbon, tmp_path):
    """The version exists so that a consumer can refuse rather than misinterpret, which is a
    claim about behaviour and is therefore tested as one."""
    path = carbon.write(tmp_path / "carbon.scp")
    path.write_text("".join(
        "format_version  {}\n".format(FORMAT_VERSION + 1)
        if line.startswith("format_version") else line
        for line in path.read_text().splitlines(True)))

    with pytest.raises(ValueError, match="refusing to guess"):
        read_parameters(path)


def test_the_file_is_written_whole_or_not_at_all(carbon, tmp_path):
    """Written to a temporary name and moved into place, as every stored product here is: a
    file truncated by an interrupt is worse than no file, because it parses."""
    path = carbon.write(tmp_path / "sub" / "carbon.scp")

    assert path.exists() and path.stat().st_size > 0
    assert not list(tmp_path.glob("**/*.partial"))


# --- With spin-orbit coupling, the whole feature at once ------------------------------------

def test_the_constants_reach_the_file_with_the_record_that_qualifies_them(tmp_path):
    """One run of everything: boron, with the default screening and therefore with a
    four-component atomic solve behind it (sub-second for this element).

    What is asserted is the join between the two halves — the constant in the ``[ZETA]``
    section, in both units, with the screening record beside it in the header and in the
    provenance. The value itself is asserted where it belongs, against the hydrogenic closed
    form and against a second route through the two-component machinery.
    """
    path = tmp_path / "boron.scp"
    result = slater_condon_parameters("B", "1s2 2s2 2p1", basis=BASIS, shells=("2s", "2p"),
                                      memory_gb=MEMORY_GB, conv_tol=1e-11, file=path,
                                      report=False)
    back = read_parameters(path)

    assert result.data.soc is not None
    assert back["header"]["spin_orbit_screening"] == "x2camf"
    assert set(back["zeta"]) == {"2p"}               # the 2s shell has no constant at all
    stored = back["zeta"]["2p"]
    assert stored["zeta"] == pytest.approx(result.spin_orbit["2p"].zeta, rel=1e-12)
    assert stored["splitting_cm"] == pytest.approx(1.5 * stored["zeta_cm"], abs=1e-3)
    assert back["provenance"]["screening"]["method"] == "x2camf"
    assert np.isclose(float(back["header"]["hartree_to_cm"]), 219474.63, atol=1e-2)
