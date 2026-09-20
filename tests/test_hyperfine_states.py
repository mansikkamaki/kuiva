"""The hyperfine field in the state basis: the invariants, the report and the file.

``tests/test_hyperfine_operator.py`` pins the AO operator — the non-relativistic-limit
identity, the four-component check on one-electron ions, the hydrogen contact value. This file
picks the operator up where that one leaves it and follows it through the CI route: the
transition-density contraction, the phase-invariant reductions, and the two new pieces of the
property file, ``[NUCLEI]`` and the ``T_<k>_u`` blocks.

⚠ **What each check can fail on**, since that is how this suite is graded and not by how many
digits it agrees to:

* **Wigner-Eckart inside a free-ion ``J`` manifold** is the one with real teeth. Every vector
  operator is proportional to ``J`` there, so ``Tr_b(T_i T_j)``, ``Tr_b(mu_i T_j)`` and
  ``Tr_b(mu_i mu_j)`` are all isotropic and satisfy ``Tr(mu.T)^2 = Tr(mu.mu) Tr(T.T)`` — an
  identity nothing in this program can influence, and one that a permuted Cartesian component,
  a rotated frame or the conjugation trap of the orbital transformation law breaks while
  leaving every hermiticity and degeneracy check intact. The test rotates by a **genuinely
  complex** unitary, because a real-arithmetic check cannot see a conjugation error at all.
* **The ratio ``A(j=1/2) / A(j=3/2) = 5``** for a single ``p`` electron. Analytic
  (``A_j ~ l(l+1)/[j(j+1)]``, Abragam & Bleaney ch. 17), independent of the radial function,
  of the basis and of every convention in this program — so it fails on a wrong weighting of
  the orbital against the spin mechanism, which the AO-level identity cannot see because it is
  evaluated at one nucleus and not across two ``j`` levels.
* **Time-reversal oddness**, which says the inactive core contributes exactly nothing. ⚠ The
  hyperfine field is odd like ``L`` and ``S`` and unlike the electric dipole, so a nonzero
  inactive trace here is a statement about the orbitals; the test asserts the theorem and that
  the code checks it rather than assuming it.
* **The file**, round-tripped through :func:`kuiva.props.dump.read_dump` and
  :meth:`~kuiva.props.dump.PropertyMatrices.from_dump`, including the refusals that exist so a
  nuclear table and its operators cannot come apart.
"""
import numpy as np
import pytest

from kuiva.interface import api
from kuiva.props import dump
from kuiva.props.dump import HYPERFINE_UNIT, PropertyMatrices
from kuiva.props.multiplet import (analyse_spectrum, block_collinearity, block_cross_tensor,
                                   block_hyperfine_tensor, multiplet_hyperfine_values)
from kuiva.util.units import HARTREE_TO_MHZ

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# --- fixtures ---------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def boron_hyperfine():
    """``B 2p^1`` with the ``11``B nucleus — the analytic free-ion case, in under a second.

    The same state-averaged CASSCF ``tests/test_dump.py`` uses for the ``g = 2/3, 4/3``
    targets: six equally weighted roots restore the spherical symmetry ROHF breaks, which is
    what makes both the Lande factors and the hyperfine ratio below analytic at all. One
    electron in one ``p`` shell is also the only configuration for which the ``A_j`` ratio has
    a closed form with no radial integral in it.
    """
    mol = api.Molecule(atoms=[("B", (0.0, 0.0, 0.0))], basis="x2c-SVPall-2c", spin=1)
    reference = api.spinor_reference(mol, screening="none", memory_gb=8,
                                     hyperfine={"B": 11})
    outcome = api.casscf(reference, character=("B", "p"), n_active=6, n_active_elec=1,
                         n_states=6, mode="second-order", conv_grad=1e-6, report=False)
    assert outcome.converged
    return reference, outcome


@pytest.fixture(scope="module")
def boron_matrices(boron_hyperfine):
    reference, outcome = boron_hyperfine
    return api.property_matrices(reference, outcome)


def _sorted(matrices, op):
    """An operator re-indexed into the energy-ordered basis the multiplets are indexed in."""
    order = np.argsort(np.asarray(matrices.energies, dtype=float))
    return np.asarray(op)[:, order, :][:, :, order]


# --- the contraction reaches the state basis at all --------------------------------------------

def test_the_matrices_arrive_with_the_nuclear_table(boron_matrices):
    """Matrices and table together, because neither is usable alone: the operators name no
    nucleus and the table states no coupling."""
    m = boron_matrices
    assert m.has_hyperfine and m.hyperfine_labels == ("B1",)
    assert m.hyperfine["B1"].shape == (3, 6, 6)
    record = m.nucleus("B1")
    assert record["label"] == "11B" and record["twice_spin"] == 3
    assert record["element"] == "B" and record["atomic_number"] == 5
    assert record["g"] == pytest.approx(1.7924326)
    assert "Stone" in record["source"]
    with pytest.raises(KeyError, match="no nucleus labelled"):
        m.nucleus("B2")


def test_the_operators_are_hermitian_in_the_state_basis(boron_matrices):
    t = boron_matrices.hyperfine["B1"]
    scale = float(np.abs(t).max())
    assert scale > 0.0
    assert np.abs(t - t.conj().transpose(0, 2, 1)).max() < 1e-13 * scale


def test_the_inactive_core_contributes_exactly_nothing(boron_matrices):
    """⚠ The hyperfine field is time **odd**, like ``L`` and ``S`` and unlike ``r``: a Kramers
    pair contributes ``<psi|T|psi> + <T psi|T|T psi> = 0``. So a Kramers-paired inactive space
    carries no hyperfine coupling at all, and a nonzero value would be a statement about the
    *orbitals* rather than about the nucleus. It is computed and checked, never assumed away —
    which is why the number exists to be asserted here."""
    trace = boron_matrices.hyperfine_inactive["B1"]
    scale = float(np.abs(boron_matrices.hyperfine["B1"]).max())
    assert np.abs(trace).max() < 1e-10 * scale


def test_a_broken_inactive_space_warns_rather_than_being_dropped(kuiva_caplog):
    """The guard that can fire: :func:`kuiva.props.dump.inactive_moment` is called with
    ``expect_zero=True`` for ``T``, so an inactive set that is no longer Kramers paired is
    reported instead of silently contributing a term nobody expected."""
    op = np.zeros((3, 4, 4), dtype=complex)
    op[2] = np.diag([1.0, 1.0, 0.0, 0.0])                # both partners the same sign: broken
    trace = dump.inactive_moment(op, [0, 1], name="T(Fe1)", expect_zero=True)
    assert trace[2] == pytest.approx(2.0)
    assert any("T(Fe1)" in r.message and "Kramers-paired" in r.message
               for r in kuiva_caplog.records)


def test_the_inactive_guard_fires_at_hyperfine_magnitudes(kuiva_caplog):
    """⚠ **A guard that cannot fire proves nothing, and this one nearly could not.**

    ``L`` and ``S`` are of order one in hbar, so the module's absolute inactive tolerance
    (1e-08) sits far above a congruence's rounding and far below any real moment. The
    hyperfine field is of order **1e-08 Eh per nuclear magneton**, so the same absolute number
    sits four orders *above* the whole operator: a completely broken inactive space would have
    passed in silence. The tolerance is therefore relative for ``T``, and this drives the
    warning from an operator of realistic magnitude — which the absolute reading does not.

    The scale below is a **light** nucleus's, two orders under the tolerance, which is the
    case that makes the point: the inactive block is 100% broken and the absolute reading
    still sees a number smaller than its own threshold.
    """
    scale = 1.0e-10
    op = np.zeros((3, 4, 4), dtype=complex)
    op[2] = scale * np.diag([1.0, 1.0, 0.0, 0.0])        # both partners alike: not a pair
    properties = _FakeHyperfineProperties(op)
    coeff = np.eye(4, dtype=complex)
    tdm = np.zeros((1, 1, 2, 2), dtype=complex)
    states, traces, nuclei, record = dump._hyperfine_states(
        coeff, properties, np.array([2, 3]), np.array([0, 1]), tdm,
        inactive_tol=dump.DEFAULT_INACTIVE_TOL)
    assert traces["Fe1"][2] == pytest.approx(2.0 * scale)
    assert states["Fe1"].shape == (3, 1, 1)
    assert nuclei[0]["atom_label"] == "Fe1" and record["unit"] == HYPERFINE_UNIT
    assert any("T(Fe1)" in r.message and "Kramers-paired" in r.message
               for r in kuiva_caplog.records)
    # ...and the absolute reading of the same tolerance would have said nothing.
    kuiva_caplog.clear()
    dump.inactive_moment(op, [0, 1], name="T(Fe1)", expect_zero=True,
                         tol=dump.DEFAULT_INACTIVE_TOL)
    assert not kuiva_caplog.records


class _FakeHyperfineProperties:
    """One nucleus with a chosen operator, in the shape ``_hyperfine_states`` consumes.

    ⚠ It stands in for the *front end*, not for anything under test: what runs is
    :func:`kuiva.props.dump._hyperfine_states` itself, on an operator whose inactive block is
    deliberately not Kramers paired — which no converged calculation cheap enough for this
    suite would produce.
    """

    class _Nucleus:
        def __init__(self, operator):
            self.operator, self.label = operator, "Fe1"

    class _Integrals:
        def __init__(self, operator):
            self.nuclei = (_FakeHyperfineProperties._Nucleus(operator),)

        def provenance(self):
            return {"unit": HYPERFINE_UNIT, "nuclei": [{"atom_label": "Fe1", "label": "57Fe"}]}

    def __init__(self, operator):
        self._integrals = self._Integrals(operator)
        self.has_hyperfine = True

    def hyperfine_integrals(self):
        return self._integrals


# --- Tier 0, item 4: Wigner-Eckart on the free-ion manifolds -----------------------------------

def test_the_invariants_are_isotropic_inside_a_free_ion_manifold(boron_matrices):
    """Inside one ``2J+1`` block every vector operator is proportional to ``J``, so all three
    invariant tensors are multiples of the identity. ⚠ A permuted or rotated component would
    keep ``Tr(T.T)`` and break this."""
    m = boron_matrices
    t = _sorted(m, m.hyperfine["B1"])
    mu = _sorted(m, m.mu)
    for block in m.analyse():
        for tensor in (block_hyperfine_tensor(t, block.start, block.size),
                       block_cross_tensor(mu, t, block.start, block.size)):
            diag = np.diag(tensor)
            scale = float(np.abs(diag).max())
            assert scale > 0.0
            # ⚠ A relative bound, and a loose one: the isotropy is exact by symmetry and the
            # residual here is the CASSCF's own convergence (~1e-8 relative), not the
            # arithmetic's. What it has to separate is a *permuted* component, which would sit
            # at O(1).
            assert np.abs(tensor - np.diag(diag)).max() < 1e-6 * scale
            assert diag == pytest.approx([diag[0]] * 3, rel=1e-6)


def test_the_hyperfine_field_is_collinear_with_the_moment(boron_matrices):
    """``Tr(mu.T)^2 = Tr(mu.mu) Tr(T.T)`` — the Cauchy-Schwarz equality, which holds exactly
    when ``T`` is a real multiple of ``mu`` on the block. Wigner-Eckart guarantees it inside a
    free-ion ``J`` manifold, and nothing in the quadratic invariants can see it."""
    m = boron_matrices
    t = _sorted(m, m.hyperfine["B1"])
    mu = _sorted(m, m.mu)
    for block in m.analyse():
        c = block_collinearity(mu, t, block.start, block.size)
        assert abs(abs(c) - 1.0) < 1e-10, block.size
        # Both levels of a p^1 ion have positive A and positive g, so the sign is the same on
        # each -- the relative sign is what this invariant carries and |A| cannot.
        assert c < 0.0


def test_the_invariants_survive_an_arbitrary_complex_rotation_inside_a_block(boron_matrices):
    """⚠ **The test has to rotate by a genuinely complex unitary**, because the two
    transformation laws a density and an operator obey coincide in real arithmetic: a
    conjugation error is invisible to any real-valued check. Degenerate states mix arbitrarily
    in exactly this way, which is why an element of ``T`` means nothing and these traces mean
    everything.
    """
    m = boron_matrices
    t = _sorted(m, m.hyperfine["B1"])
    mu = _sorted(m, m.mu)
    blocks = m.analyse()
    rng = np.random.default_rng(3)
    rotated_t, rotated_mu = t.copy(), mu.copy()
    for block in blocks:
        n = block.size
        a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        u = np.linalg.qr(a)[0]
        assert np.abs(u.imag).max() > 0.1, "the rotation must be genuinely complex"
        sl = slice(block.start, block.start + n)
        for op in (rotated_t, rotated_mu):
            op[:, sl, sl] = np.stack([u.conj().T @ ok[sl, sl] @ u for ok in op])

    for block in blocks:
        before = block_cross_tensor(mu, t, block.start, block.size)
        after = block_cross_tensor(rotated_mu, rotated_t, block.start, block.size)
        assert np.abs(after - before).max() < 1e-10 * float(np.abs(before).max())
        assert (block_collinearity(rotated_mu, rotated_t, block.start, block.size)
                == pytest.approx(block_collinearity(mu, t, block.start, block.size),
                                 abs=1e-12))


def test_the_single_p_electron_ratio_of_hyperfine_constants_is_five(boron_matrices):
    """⚠ **The analytic target, and the only check here that spans two ``j`` levels.**

    For one electron in an ``nl`` shell the magnetic hyperfine constant is
    ``A_j = a_l l(l+1)/[j(j+1)]`` (Abragam & Bleaney ch. 17), with the radial integral ``a_l``
    common to both levels. For ``l = 1`` that is ``A(1/2)/A(3/2) = (8/3)/(8/15) = 5``, exactly,
    with no basis-set error and nothing from this program in it.

    It fails on a wrong relative weight of the orbital and spin mechanisms — which the AO-level
    identity cannot see, being evaluated at one nucleus on one operator — and on a wrong ``J``
    normalization in the reduction. The observed 0.15% excess is the relativistic correction to
    the ratio at ``Z = 5``.
    """
    m = boron_matrices
    t = _sorted(m, m.hyperfine["B1"])
    g_n = float(m.nucleus("B1")["g"])
    blocks = m.analyse()
    assert [b.size for b in blocks] == [2, 4], "not the 2P1/2 / 2P3/2 pattern"
    a = [multiplet_hyperfine_values(block_hyperfine_tensor(t, b.start, b.size), b.size, g_n)
         for b in blocks]
    for values in a:
        assert values == pytest.approx([values[0]] * 3, rel=1e-6)
    assert a[0][0] / a[1][0] == pytest.approx(5.0, rel=5e-3)
    # ⚠ A magnitude band, and it is not an accuracy claim: what it catches is a factor of two,
    # a missing nuclear magneton or a Hartree written as a wavenumber -- errors that leave the
    # *ratio* above untouched, since it divides them out. How close this is to experiment is a
    # question about the active space (see the core-polarization warning), not about the code.
    assert 300.0 < a[0][0] < 380.0


def test_the_reduction_scales_with_the_nuclear_g_factor(boron_matrices):
    """``|A|`` is linear in ``g_N`` and the stored operator does not contain it — which is the
    whole point of writing the field: a consumer changes the isotope without Kuiva."""
    m = boron_matrices
    t = _sorted(m, m.hyperfine["B1"])
    block = m.analyse()[0]
    tensor = block_hyperfine_tensor(t, block.start, block.size)
    one = multiplet_hyperfine_values(tensor, block.size, 1.0)
    twice = multiplet_hyperfine_values(tensor, block.size, -2.0)
    assert twice == pytest.approx([2.0 * x for x in one])     # magnitude, so the sign is lost
    assert multiplet_hyperfine_values(tensor, 1, 1.0) == ()    # a singlet carries no A tensor


def test_a_hyperfine_matrix_is_scaled_to_megahertz_by_the_unit_table(boron_matrices):
    """The one conversion between the stored operator and a number a spectroscopist reads,
    asserted against the unit table rather than against a literal."""
    m = boron_matrices
    t = _sorted(m, m.hyperfine["B1"])
    block = m.analyse()[0]
    tensor = block_hyperfine_tensor(t, block.start, block.size)
    in_hartree = multiplet_hyperfine_values(tensor, block.size, 1.0)[0] / HARTREE_TO_MHZ
    # H_hf = a I.S on a doublet, so a = 2 g_N <T_z> in Eh; the reduction must reproduce that
    # from the block trace alone.
    assert in_hartree == pytest.approx(
        np.sqrt(3.0 * np.diag(tensor)[0] / (0.5 * 1.5 * 2.0)), rel=1e-12)


# --- the spectrum analysis carries them through ------------------------------------------------

def test_analyse_fills_the_per_nucleus_tensors(boron_matrices):
    m = boron_matrices
    blocks = m.analyse()
    for block in blocks:
        assert set(block.hyperfine) == {"B1"}
        assert block.hyperfine["B1"].shape == (3, 3)
        assert set(block.hyperfine_cross) == {"B1"}


def test_analyse_without_the_moment_leaves_the_mixed_invariant_absent(boron_matrices):
    """⚠ ``Tr_b(mu_i T_j)`` needs both operators, and "not computed" is not "zero": a mixed
    tensor of zeros would read as two orthogonal operators."""
    m = boron_matrices
    blocks = analyse_spectrum(m.energies, mu=None, hyperfine=m.hyperfine)
    assert all(b.hyperfine is not None and b.hyperfine_cross is None for b in blocks)


def test_hyperfine_changes_no_existing_number(boron_matrices):
    """The blocking and every moment invariant are the energies' and ``mu``'s decision alone,
    so passing the hyperfine matrices moves nothing that was there before."""
    m = boron_matrices
    with_hf = m.analyse()
    without = analyse_spectrum(m.energies, m.mu, d=m.d)
    assert [b.size for b in with_hf] == [b.size for b in without]
    for a, b in zip(with_hf, without):
        assert a.g_values == b.g_values
        assert np.array_equal(a.m_tensor, b.m_tensor)


# --- the report --------------------------------------------------------------------------------

def test_the_report_prints_the_reduction_and_stays_ascii(boron_matrices, kuiva_caplog):
    """The output stream is ASCII only and the table names what it is showing — a reader has to
    be able to tell an ``|A|`` reduction from a fitted tensor, because Kuiva writes no tensor.
    """
    boron_matrices.report()
    text = kuiva_caplog.text
    assert "Hyperfine field operators" in text
    assert "|A_1| [MHz]" in text and "A.g" in text
    assert "B1" in text and "11B" in text
    for record in kuiva_caplog.records:
        record.getMessage().encode("ascii")


# --- the file ----------------------------------------------------------------------------------

def test_the_dump_carries_the_operators_and_the_table(boron_matrices, tmp_path):
    """Written and read back element for element, table included. A format nobody reads back
    has an undetected ambiguity in it, and a whitespace-delimited table is exactly where one
    would hide."""
    m = boron_matrices
    back = dump.read_dump(m.write(tmp_path / "b.prop", title="hyperfine"))
    for k, axis in enumerate("xyz"):
        assert np.array_equal(back["matrices"]["T_B1_" + axis], m.hyperfine["B1"][k])
    assert len(back["nuclei"]) == 1
    record = back["nuclei"][0]
    assert record["atom_label"] == "B1" and record["label"] == "11B"
    assert record["atom"] == 1 and record["atomic_number"] == 5
    assert record["twice_spin"] == 3
    assert record["g"] == pytest.approx(m.nucleus("B1")["g"], rel=0, abs=0)
    assert record["quadrupole_barn"] == pytest.approx(0.040591)
    assert record["position_bohr"] == [0.0, 0.0, 0.0]
    assert np.allclose(back["inactive"]["T(B1)"], m.hyperfine_inactive["B1"], atol=1e-12)


def test_the_header_states_every_approximation(boron_matrices, tmp_path):
    """⚠ Three approximations appear in no number in the file: the unperturbed X2C
    transformation, the absent two-electron picture change and the nuclear magnetization
    model. And the picture change is stated **unconditionally**, because for this operator
    family it is not a choice — which is the recorded exception to one flag governing both
    property operators."""
    header = dump.read_dump(boron_matrices.write(tmp_path / "b.prop"))["header"]
    assert header["hyperfine_unit"] == HYPERFINE_UNIT
    assert header["hyperfine_operator"].startswith("H_hf =")
    assert "always applied" in header["hyperfine_picture_change"]
    assert header["hyperfine_decoupling"] == "1e"
    assert header["hyperfine_nuclear_model"] == "point"
    assert "unperturbed" in header["hyperfine_x2c_response"]
    assert "not applied" in header["hyperfine_2e_picture_change"]
    assert float(header["hyperfine_g_electron"]) == pytest.approx(2.00231930436, rel=1e-11)
    assert float(header["nuclear_magneton_au"]) > 0.0
    assert header["n_hyperfine_nuclei"] == "1"
    # ⚠ mu and d stay bare here; the two families are stated separately and that is the point.
    assert header["picture_change_on_properties"] == "none"


def test_the_dump_warns_about_what_no_number_in_it_shows(boron_matrices, tmp_path,
                                                         kuiva_caplog):
    """The standing obligation, discharged every time the file is written: the treatment of the
    operators and the core-polarization limitation the active space cannot repair."""
    boron_matrices.write(tmp_path / "b.prop")
    assert any("HYPERFINE FIELD operators" in r.message and "core-s spin polarization"
               in r.message for r in kuiva_caplog.records)
    text = (tmp_path / "b.prop").read_text()
    assert "H_hf = sum_k g_N(k) sum_u T_<k>_u (x) I_<k>_u" in text
    assert "ISOTOPE-INDEPENDENT" in text


def test_the_dump_round_trips_through_from_dump(boron_matrices, tmp_path):
    """⚠ Bitwise on the matrices, because the comparison this exists for would otherwise be a
    comparison of two different things — and through the file's own two sections, not through
    the provenance JSON, since a consumer that parses no JSON must still get the nuclei."""
    m = boron_matrices
    back = PropertyMatrices.from_dump(m.write(tmp_path / "b.prop"))
    assert back.has_hyperfine and back.hyperfine_labels == ("B1",)
    assert np.array_equal(back.hyperfine["B1"], m.hyperfine["B1"])
    assert np.allclose(back.hyperfine_inactive["B1"], m.hyperfine_inactive["B1"], atol=1e-12)
    assert back.nucleus("B1")["label"] == "11B"
    assert back.hyperfine_record["nuclear_model"] == "point"
    assert back.hyperfine_record["unit"] == HYPERFINE_UNIT
    # ...and the deliverable: the reduction agrees.
    for a, b in zip(back.analyse(), m.analyse()):
        assert np.allclose(a.hyperfine["B1"], b.hyperfine["B1"], rtol=0, atol=0)
        assert np.allclose(a.hyperfine_cross["B1"], b.hyperfine_cross["B1"], rtol=0, atol=0)


def test_a_file_without_hyperfine_comes_back_with_none_not_zeros(tmp_path):
    """⚠ "Not computed" and "computed and small" are different statements, and a whole active
    space can be the difference between them. The same rule the electric dipole follows."""
    rng = np.random.default_rng(2)
    n = 4
    a = rng.normal(size=(3, n, n)) + 1j * rng.normal(size=(3, n, n))
    mu = np.stack([0.5 * (m + m.conj().T) for m in a])
    original = PropertyMatrices(energies=np.array([-1.0, -1.0, -0.5, -0.5]),
                                mu=mu, l=mu, s=mu)
    back = PropertyMatrices.from_dump(original.write(tmp_path / "plain.prop"))
    assert back.hyperfine is None and not back.has_hyperfine
    assert back.hyperfine_nuclei == () and back.hyperfine_record == {}
    text = (tmp_path / "plain.prop").read_text()
    assert "[NUCLEI]" not in text and "hyperfine_unit" not in text


def test_a_nuclear_table_without_its_operators_is_refused(boron_matrices, tmp_path):
    """The two halves of the contract cannot come apart: a table naming a nucleus whose
    matrices are absent describes a coupling the file cannot supply."""
    path = boron_matrices.write(tmp_path / "b.prop")
    text = path.read_text()
    start = text.index("[MATRIX T_B1_x]")
    end = text.index("[MATRIX L_x]")
    path.write_text(text[:start] + text[end:])
    with pytest.raises(ValueError, match="no complete set of"):
        PropertyMatrices.from_dump(path)


def test_a_shifted_nuclei_row_is_refused_rather_than_misread(boron_matrices, tmp_path):
    """⚠ A whitespace-delimited table misparses into plausible numbers if a column moves, so
    the row's field count is checked rather than trusted."""
    path = boron_matrices.write(tmp_path / "b.prop")
    lines = path.read_text().splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.lstrip().startswith("0 ") or line.startswith("    0     1"):
            lines[i] = line.replace("B1", "B1 extra", 1)
            break
    else:                                                    # pragma: no cover - guard
        pytest.fail("no [NUCLEI] data row found in the file")
    path.write_text("".join(lines))
    with pytest.raises(ValueError, match="fields before"):
        dump.read_dump(path)


def test_a_header_count_that_disagrees_with_the_table_is_refused(boron_matrices, tmp_path):
    """A T matrix matched to the wrong nucleus is Hermitian, plausible and wrong."""
    path = boron_matrices.write(tmp_path / "b.prop")
    path.write_text(path.read_text().replace("n_hyperfine_nuclei               1",
                                             "n_hyperfine_nuclei               2"))
    with pytest.raises(ValueError, match="hyperfine nuclei"):
        dump.read_dump(path)


def test_the_stage_writes_them_with_no_second_switch(tmp_path):
    """⚠ Naming the nuclei at ingestion **is** the request: there is no flag on
    :class:`kuiva.PropertyDump` that could throw away an operator already paid for, or leave a
    file that computed the coupling and did not say so. Run through the class API — the primary
    user surface — because that is where the plumbing between ``hyperfine=`` and the file is.
    """
    import kuiva

    scf = kuiva.ScalarSCF(kuiva.Molecule(atoms=[("B", (0.0, 0.0, 0.0))],
                                         basis="x2c-SVPall-2c", spin=1),
                          screening="none", memory_gb=8, hyperfine={"B": 11}).run()
    reference = kuiva.Reference(scf).run()
    cas = kuiva.CASSCF(reference, character=("B", "p"), n_active=6, n_active_elec=1,
                       n_states=6, mode="second-order", conv_grad=1e-6, report=False).run()
    stage = kuiva.PropertyDump(cas, tmp_path / "stage.prop", report=False).run()
    assert stage.matrices.has_hyperfine
    assert "T_B1_z" in dump.read_dump(stage.path)["matrices"]
    assert "hyperfine nuclei" in stage.summary() and "B1" in stage.summary()
