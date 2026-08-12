"""Tests for the molden spinor-density dump.

The format is a contract with software this project does not control, so it is validated
against an independent implementation of the same format — PySCF's ``molden.from_mo`` — and
by round-tripping through a parser rather than by inspecting our own strings. Agreement is
**exact**: same basis, same ordering, same coefficients.

What each group of tests can fail on:

* **Format equality** catches the AO ordering, the ``[GTO]`` contraction layout and the
  spherical markers. This is bookkeeping against another program's bookkeeping.
* **The density invariants** catch the physics: occupations must sum to the electron count and
  the components must reproduce the density (that last is asserted on a grid in
  ``test_spinor_density.py``, which is where the decomposition itself lives).
* **The high-l tests** cover the two required behaviours — drop with a WARNING by default,
  write with an ``[11h]`` marker on request — and the measurement of what a dropped function
  costs, which is the thing that keeps a truncated picture honest.
"""
import numpy as np
import pytest

from kuiva.interface.api import Molecule
from kuiva.interface.pyscf_bridge import ao_layout, run_scalar_x2c
from kuiva.props.molden import MoldenOrbital, write_molden, write_spinor_molden
from kuiva.spinor.expand import expand_scalar_mos, time_reverse


@pytest.fixture(scope="module")
def ticl():
    from pyscf import gto
    return gto.M(atom=[["Ti", (0, 0, 0)], ["Cl", (0, 0, 2.3)]], basis="x2c-SVPall-2c",
                 spin=1, verbose=0)


@pytest.fixture(scope="module")
def cerium():
    """A basis that reaches g (l = 4) — the highest the molden standard defines."""
    from pyscf import gto
    return gto.M(atom=[["Ce", (0, 0, 0)]], basis="x2c-SVPall-2c", charge=3, spin=1, verbose=0)


# --- The format, against an independent implementation --------------------------------------

def test_file_is_identical_to_pyscfs_writer(ticl, tmp_path):
    """The load-bearing format test: same geometry, same basis, same coefficients, after both
    files have gone through PySCF's own parser."""
    from pyscf.tools import molden

    lay = ao_layout(ticl)
    rng = np.random.default_rng(0)
    mo = np.linalg.qr(rng.standard_normal((ticl.nao, ticl.nao)))[0]
    ene = np.arange(ticl.nao) * 0.1
    occ = np.zeros(ticl.nao)
    occ[:5] = 1.0

    ours, theirs = tmp_path / "ours.molden", tmp_path / "theirs.molden"
    write_molden(ours, lay, [MoldenOrbital(mo[:, i], occ[i], ene[i], "A")
                             for i in range(ticl.nao)])
    molden.from_mo(ticl, str(theirs), mo, ene=ene, occ=occ)

    a, b = molden.load(str(ours)), molden.load(str(theirs))
    assert a[0].nao == b[0].nao == ticl.nao
    assert a[0]._basis == b[0]._basis
    assert np.max(np.abs(a[0].atom_coords() - b[0].atom_coords())) < 1e-12
    assert np.max(np.abs(a[2] - b[2])) == 0.0            # coefficients, exactly
    assert np.allclose(a[1], b[1]) and np.allclose(a[3], b[3])


def test_g_functions_survive_the_round_trip(cerium, tmp_path):
    """``[9g]`` is the top of the standard and the project default basis reaches it on a
    lanthanide, so this is the case that must not be lost."""
    from pyscf.tools import molden

    lay = ao_layout(cerium)
    assert lay.max_l == 4
    mo = np.linalg.qr(np.random.default_rng(1).standard_normal((cerium.nao, cerium.nao)))[0]
    path = tmp_path / "ce.molden"
    info = write_molden(path, lay, [MoldenOrbital(mo[:, i], 0.0, 0.0)
                                    for i in range(4)])
    assert info["dropped_ao"] == 0
    parsed = molden.load(str(path))
    assert parsed[0].nao == cerium.nao
    # The file is text with 14 significant digits (the format PySCF uses too), so a round trip
    # is exact only to that. File *against file* is bitwise — that is the test above.
    assert np.max(np.abs(parsed[2] - mo[:, :4])) < 1e-13


def test_coefficient_count_is_checked(ticl, tmp_path):
    lay = ao_layout(ticl)
    with pytest.raises(ValueError, match="coefficients for"):
        write_molden(tmp_path / "x.molden", lay, [MoldenOrbital(np.zeros(3), 1.0, 0.0)])


# --- High-l handling (warn and drop by default, opt in to keep) -------------------------

def make_high_l_layout(lay):
    """A layout with one artificial h shell appended to the first atom."""
    from kuiva.basis.layout import Shell, build_layout

    shells = list(lay.shells) + [Shell(atom=0, l=5, exponents=np.array([1.5]),
                                       coefficients=np.array([1.0]))]
    labels = list(lay.ao_labels) + ["9h{:+d}".format(m) for m in range(-5, 6)]
    return build_layout(lay.atom_symbols, lay.atom_charges, lay.coords_bohr, shells, labels)


def test_high_l_dropped_by_default_with_a_warning(ticl, tmp_path, kuiva_caplog):
    lay = make_high_l_layout(ao_layout(ticl))
    info = write_molden(tmp_path / "drop.molden", lay,
                        [MoldenOrbital(np.ones(lay.nao), 1.0, 0.0)])
    assert info["dropped_ao"] == 11                        # one h shell
    assert info["n_ao_written"] == lay.nao - 11
    assert any("dropping" in r.getMessage() and "l > 4" in r.getMessage()
               for r in kuiva_caplog.records)
    text = (tmp_path / "drop.molden").read_text()
    assert "[11h]" not in text and "[9g]" in text


def test_high_l_written_on_request_with_a_warning(ticl, tmp_path, kuiva_caplog):
    """⚠ Non-standard by construction — the WARNING and the marker are both required."""
    lay = make_high_l_layout(ao_layout(ticl))
    info = write_molden(tmp_path / "keep.molden", lay,
                        [MoldenOrbital(np.ones(lay.nao), 1.0, 0.0)], include_high_l=True)
    assert info["dropped_ao"] == 0
    assert info["n_ao_written"] == lay.nao
    assert any("NOT standard" in r.getMessage() for r in kuiva_caplog.records)
    text = (tmp_path / "keep.molden").read_text()
    assert "[11h]" in text
    assert " h    1 1.00" in text                          # the shell itself is in [GTO]


def test_high_l_ordering_follows_moldens_rule(ticl):
    """``m = 0, +1, -1, ...`` continued past g — the interpretation the option commits to."""
    from kuiva.basis.layout import molden_ao_order, molden_m_order

    lay = make_high_l_layout(ao_layout(ticl))
    assert molden_m_order(5) == [5, 6, 4, 7, 3, 8, 2, 9, 1, 10, 0]
    order = molden_ao_order(lay.shells, max_l=None)
    assert order.size == lay.nao
    assert np.array_equal(np.sort(order), np.arange(lay.nao))


# --- Spinor densities ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def water():
    mol = Molecule.from_xyz_string("O 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.24",
                                   basis="x2c-SVPall-2c")
    data = run_scalar_x2c(mol, screening="none", memory_gb=4.0)
    sb = expand_scalar_mos(data.mo_coeff, data.mo_energy, data.mo_occ, basis="ao")
    return data, sb


def test_written_occupations_are_the_electron_count(water, tmp_path):
    """The invariant that makes the file's total density the true one."""
    data, sb = water
    rep = write_spinor_molden(tmp_path / "h2o.molden", data.ao_layout, sb.c, data.s_ao,
                              occupation=sb.occ, energy=sb.energy)
    assert rep.written_electrons == pytest.approx(float(sb.occ.sum()), abs=1e-10)
    assert rep.written_electrons == pytest.approx(10.0, abs=1e-10)


def test_a_scalar_reference_needs_one_component_per_pair(water, tmp_path):
    """The SOC-free guess is real, so each Kramers pair is a single real orbital — and the
    file then *is* the familiar MO picture."""
    data, sb = water
    rep = write_spinor_molden(tmp_path / "h2o.molden", data.ao_layout, sb.c, data.s_ao,
                              occupation=sb.occ, columns=range(10))
    assert rep.n_groups == 5
    assert rep.n_orbitals == 5
    assert np.all(rep.n_components == 1)
    assert np.allclose(rep.leading_weight, 1.0, atol=1e-12)


def test_spin_orbit_mixing_produces_two_components(water, tmp_path):
    """A spinor mixing two spatial orbitals with a relative phase of i needs two pictures."""
    data, sb = water
    nao = data.s_ao.shape[0]
    c = np.zeros((2 * nao, 2), dtype=complex)
    c[:nao, 0] = (sb.c[:nao, 0] + 1j * sb.c[:nao, 2]) / np.sqrt(2.0)
    c[:, 1] = time_reverse(c[:, :1])[:, 0]
    rep = write_spinor_molden(tmp_path / "mixed.molden", data.ao_layout, c, data.s_ao,
                              occupation=np.ones(2))
    assert rep.n_components[0] == 2
    assert rep.leading_weight[0] == pytest.approx(0.5, abs=1e-8)


def test_kramers_grouping_halves_the_file(water, tmp_path):
    data, sb = water
    paired = write_spinor_molden(tmp_path / "a.molden", data.ao_layout, sb.c, data.s_ao,
                                 occupation=sb.occ, columns=range(10), group="kramers")
    single = write_spinor_molden(tmp_path / "b.molden", data.ao_layout, sb.c, data.s_ao,
                                 occupation=sb.occ, columns=range(10), group="none")
    assert paired.n_groups * 2 == single.n_groups
    # Same physics either way: the represented electron count cannot depend on the grouping.
    assert paired.written_electrons == pytest.approx(single.written_electrons, abs=1e-10)


def test_written_file_parses_and_has_the_expected_orbital_count(water, tmp_path):
    from pyscf.tools import molden

    data, sb = water
    path = tmp_path / "h2o.molden"
    rep = write_spinor_molden(path, data.ao_layout, sb.c, data.s_ao, occupation=sb.occ,
                              columns=range(10))
    parsed = molden.load(str(path))
    assert parsed[2].shape == (data.nao, rep.n_orbitals)
    assert np.allclose(parsed[3].sum(), rep.written_electrons, atol=1e-4)


def test_header_states_what_the_file_contains(water, tmp_path):
    """⚠ The file must say it holds density components, not orbitals: it is read by people and
    programs that did not run the calculation."""
    data, sb = water
    path = tmp_path / "h2o.molden"
    write_spinor_molden(path, data.ao_layout, sb.c, data.s_ao, occupation=sb.occ,
                        columns=range(4), provenance=["method: X2C-1e"])
    text = path.read_text()
    assert "density" in text and "not orbitals" in text
    assert "phases are meaningless" in text
    assert "method: X2C-1e" in text
    assert text.startswith("[Molden Format]")


def test_truncated_weight_is_measured(water, tmp_path):
    """Dropping high-l functions truncates the plotted orbital, and the report says by how
    much rather than leaving it to be discovered."""
    data, sb = water
    lay = make_high_l_layout(data.ao_layout)
    nao = lay.nao
    c = np.zeros((2 * nao, 2), dtype=complex)
    c[:data.nao, :2] = sb.c[:data.nao, :2]
    c[nao:nao + data.nao, :2] = sb.c[data.nao:2 * data.nao, :2]
    s = np.eye(nao)
    s[:data.nao, :data.nao] = data.s_ao
    rep = write_spinor_molden(tmp_path / "trunc.molden", lay, c, s, occupation=np.ones(2))
    assert rep.dropped_ao == 11
    assert rep.truncated_weight.shape == (1,)
    assert np.all(rep.truncated_weight >= 0.0)


def test_non_degenerate_grouping_warns(water, tmp_path, kuiva_caplog):
    """A group's density is scaled by *one* occupation, so the group must be degenerate — the
    same rule the population module enforces, through the same check."""
    data, sb = water
    occ = np.array(sb.occ, dtype=float)
    occ[1] = 0.0                                       # break a pair
    write_spinor_molden(tmp_path / "bad.molden", data.ao_layout, sb.c, data.s_ao,
                        occupation=occ, columns=range(4), report=False)
    assert any("degenerate block" in r.getMessage() for r in kuiva_caplog.records)


def test_report_logs_the_component_summary(water, tmp_path, kuiva_caplog):
    data, sb = water
    write_spinor_molden(tmp_path / "h2o.molden", data.ao_layout, sb.c, data.s_ao,
                        occupation=sb.occ, columns=range(4), report=True)
    text = "\n".join(r.getMessage() for r in kuiva_caplog.records)
    assert "Molden spinor-density dump" in text
    assert "electrons represented" in text
