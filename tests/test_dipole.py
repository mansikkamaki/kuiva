"""The electric dipole operator, its assembly into state matrices, and its invariants.

The dipole rides on the same transition-density contraction the magnetic moment does, so the
arithmetic is not where it goes wrong. What goes wrong is the *assembly*, and every way of
getting it wrong leaves a Hermitian matrix of an entirely plausible size:

* drop the **nuclear** term and every transition dipole stays exactly right while the diagonal
  silently stops being a dipole moment;
* drop the **inactive** term and the valence dipole wears the name of the total — and unlike
  ``L`` and ``S`` that term is *not* zero, because ``r`` is time even;
* add the nuclear term **off** the diagonal and every pair of states acquires a spurious
  transition dipole proportional to the nuclear charge;
* get the **sign** wrong on either half and the total is still a vector of the right magnitude.

So the tests here are chosen for what they can fail on, and three of them can fail on all four
mistakes at once: a centrosymmetric molecule's dipole is zero only if all the terms are present
and correctly signed, a neutral molecule's dipole is origin independent only under the same
condition, and the free ion's parity selection rule is exact.

⚠ **The decisive one is** :func:`test_transition_dipole_matches_an_independent_pyscf_pipeline`:
its two sides share no implementation above ``libcint``. An in-house check of an assembly cannot
see an error in that assembly, however many digits it agrees to.

⚠ **Comparison goes through the phase-invariant reductions only** —
``Tr_block(d_i d_j)`` and the block-to-block line strength ``sum |d_IJ|^2``. Degenerate states
mix arbitrarily, so an element-by-element comparison of ``d`` compares arbitrary phases.
"""
import numpy as np
import pytest

from kuiva.interface import api
from kuiva.interface.pyscf_bridge import (gauge_origin_for, ingest_property_integrals,
                                          nuclear_dipole, property_integral_memory_gb)
from kuiva.props import dump
from kuiva.props.multiplet import block_line_strengths, block_dipole_tensor

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

BASIS = "x2c-SV(P)all-2c"


# --- fixtures ---------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def lih_mol():
    from pyscf import gto
    return gto.M(atom="Li 0 0 0; H 0 0 1.6", basis=BASIS, verbose=0, unit="Angstrom")


def _lih(charge=0, spin=0, n_states=6, gauge_origin=None):
    """CAS(2 electrons, 2 spatial orbitals) on LiH at the SCF orbitals — seconds, not minutes.

    A CASCI rather than a CASSCF on purpose: the orbitals are then PySCF's own converged
    sfx2c1e RHF orbitals, which is what lets an independent PySCF pipeline define the *same*
    calculation and makes the cross-check a comparison of the property assembly alone.
    """
    mol = api.Molecule(atoms=[("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.6))], basis=BASIS,
                       charge=charge, spin=spin)
    ref = api.spinor_reference(mol, screening="none", memory_gb=8,
                               gauge_origin=gauge_origin)
    n_occ = int((np.asarray(ref.data.mo_occ) > 0).sum())
    # Two spatial orbitals about the Fermi level -> the contiguous spinor range [2m, 2n).
    first = 2 * (n_occ - 1)
    active = list(range(first, first + 4))
    n_elec = 2 - int(charge)
    result = api.casci(ref, active=active, n_active_elec=n_elec, n_states=n_states,
                       report=False)
    return ref, result


@pytest.fixture(scope="module")
def lih():
    return _lih()


@pytest.fixture(scope="module")
def lih_matrices(lih):
    ref, result = lih
    return api.property_matrices(ref, result)


@pytest.fixture(scope="module")
def boron_matrices():
    """``B 2p^1``, state averaged over all six roots — the free-ion parity case."""
    mol = api.Molecule(atoms=[("B", (0.0, 0.0, 0.0))], basis="x2c-SVPall-2c", spin=1)
    ref = api.spinor_reference(mol, screening="none", memory_gb=8)
    outcome = api.casscf(ref, character=("B", "p"), n_active=6, n_active_elec=1, n_states=6,
                         mode="second-order", conv_grad=1e-6, report=False)
    assert outcome.converged
    return api.property_matrices(ref, outcome)


# --- Tier 0: the ingested integrals ------------------------------------------------------------

def test_the_position_integrals_are_symmetric(lih_mol):
    """``<mu|r_k|nu>`` is exactly symmetric for real Gaussians, and ``d`` is Hermitian only
    because of it. Structural, cheap, and it fails loudly if libcint's convention ever moves."""
    props = ingest_property_integrals(lih_mol, "mass")
    r = props.electric_dipole_ao()
    assert r.shape == (3, lih_mol.nao, lih_mol.nao)
    assert np.abs(r - r.transpose(0, 2, 1)).max() < 1e-14 * np.abs(r).max()


def test_the_symmetry_check_is_load_bearing(lih_mol, monkeypatch):
    """Corrupt the integral and ingestion must **raise**. Without this, "the integrals are
    symmetric" would be consistent with a check that cannot fail."""
    from kuiva.interface import pyscf_bridge

    good = pyscf_bridge._check_position_integrals

    def broken(position):
        bad = np.asarray(position).copy()
        bad[0, 0, 1] += 1.0
        return good(bad)

    monkeypatch.setattr(pyscf_bridge, "_check_position_integrals", broken)
    with pytest.raises(RuntimeError, match="not symmetric"):
        ingest_property_integrals(lih_mol, "mass")


def test_both_operators_are_taken_about_one_gauge_origin(lih_mol):
    """⚠ The failure this stops has no symptom: ``L`` about the centre of mass and ``r`` about
    the coordinate origin would give a file whose magnetic and electric halves refer to
    different frames, with every number in it perfectly reasonable.

    Checked by translating: moving the origin by ``delta`` must shift ``r`` by exactly
    ``-delta_k S`` and must shift ``L`` by the corresponding ``-delta x nabla``, and both must
    do it about the point the container reports.
    """
    delta = np.array([0.3, -0.7, 0.25])
    base = ingest_property_integrals(lih_mol, "mass")
    moved = ingest_property_integrals(lih_mol, ("bohr", *(base.gauge_origin + delta)))
    assert np.allclose(moved.gauge_origin, base.gauge_origin + delta, atol=1e-12)

    lih_mol.set_common_orig(np.zeros(3))
    s = np.asarray(lih_mol.intor_symmetric("int1e_ovlp"))
    ipovlp = np.asarray(lih_mol.intor("int1e_ipovlp", comp=3))
    for k in range(3):
        assert np.abs(moved.position[k] - (base.position[k] - delta[k] * s)).max() < 1e-12
    # L = -i (r - R_G) x nabla, so moving the origin by delta SUBTRACTS delta x nabla;
    # <i|nabla_a|j> is -int1e_ipovlp[a] for real Gaussians (one integration by parts).
    grad = -ipovlp
    for k in range(3):
        a, b = (k + 1) % 3, (k + 2) % 3
        cross = delta[a] * grad[b] - delta[b] * grad[a]
        assert np.abs(moved.irxp[k] - (base.irxp[k] - cross)).max() < 1e-11


def test_the_nuclear_dipole_is_the_definition(lih_mol):
    """``sum_A Z_A (R_A - R_G)``, against the sum written out by hand."""
    r_g = gauge_origin_for(lih_mol, "mass")[0]
    want = np.zeros(3)
    for ia in range(lih_mol.natm):
        want += lih_mol.atom_charge(ia) * (lih_mol.atom_coords()[ia] - r_g)
    assert nuclear_dipole(lih_mol, r_g) == pytest.approx(want, abs=1e-14)
    # A neutral molecule's nuclear dipole moves with the origin exactly as the electronic one
    # does, which is why the total does not.
    delta = np.array([1.0, 2.0, -0.5])
    moved = nuclear_dipole(lih_mol, r_g + delta)
    z_total = sum(lih_mol.atom_charge(ia) for ia in range(lih_mol.natm))
    assert moved == pytest.approx(want - z_total * delta, abs=1e-12)


def test_the_property_integral_sizing_is_exact():
    """⚠ Bounded on **both** sides against a real array's ``nbytes``, so a sizing function that
    grows a safety factor fails rather than passing more comfortably."""
    nao = 37
    a = np.zeros((3, nao, nao), dtype=np.float64)
    want = 2.0 * a.nbytes / 1024.0 ** 3
    got = property_integral_memory_gb(nao)
    assert got == pytest.approx(want, rel=0.0, abs=0.0)


# --- the assembly: the three terms, and the checks that need all three -------------------------

def test_a_free_ions_dipole_vanishes_by_parity(boron_matrices):
    """⚠ The analytic target. Every state of ``2p^1`` has the same parity, so **every** electric
    dipole matrix element — diagonal and transition alike — is exactly zero. No reference
    calculation is involved, and a sign error, a missing term or a wrong operator would all
    show up as a nonzero number here."""
    assert boron_matrices.has_dipole
    assert np.abs(boron_matrices.d).max() < 1e-8
    strengths = boron_matrices.line_strengths()
    assert np.abs(strengths).max() < 1e-16


def test_a_neutral_molecules_dipole_does_not_move_with_the_gauge_origin():
    """⚠ The check that needs all three terms at once, and the reason the header can be trusted.

    For a neutral molecule ``d(R_G) = d(0) - q R_G`` with ``q = 0``, so **every** element —
    diagonal included — is origin independent. It comes out that way only if the electronic,
    inactive and nuclear pieces are all present and all correctly signed: each of them moves
    with the origin on its own, and only the sum stands still.
    """
    _, at_mass = _lih(n_states=3)
    ref_mass, _ = _lih(n_states=3)
    ref_atom, at_atom = _lih(n_states=3, gauge_origin=("atom", 2))
    a = api.property_matrices(ref_mass, at_mass)
    b = api.property_matrices(ref_atom, at_atom)
    assert np.linalg.norm(a.gauge_origin - b.gauge_origin) > 1.0
    assert a.molecular_charge == 0
    # The two runs are separate CASCIs, so state phases may differ; compare the invariants and,
    # for the non-degenerate states this spectrum has, the diagonal itself.
    assert np.abs(np.diagonal(a.d, axis1=1, axis2=2)
                  - np.diagonal(b.d, axis1=1, axis2=2)).max() < 1e-8
    for k in range(3):
        assert block_dipole_tensor(a.d, 0, 1) == pytest.approx(
            block_dipole_tensor(b.d, 0, 1), abs=1e-8)


def test_a_charged_molecules_diagonal_moves_by_minus_q_R():
    """⚠ The mechanism behind the warning, asserted rather than asserted-about.

    ``d(R_G) = d(0) - q R_G`` on the **diagonal**; transition elements between distinct states
    carry no nuclear or inactive term at all and do not move. That is exactly why the dump warns
    and records the charge and the origin instead of refusing: a charged system's transition
    dipoles are perfectly well defined, and its permanent moments are a statement about a point.
    """
    ref_a, res_a = _lih(charge=1, spin=1, n_states=2)
    ref_b, res_b = _lih(charge=1, spin=1, n_states=2, gauge_origin=("atom", 2))
    a = api.property_matrices(ref_a, res_a)
    b = api.property_matrices(ref_b, res_b)
    assert a.molecular_charge == 1 and a.dipole_is_origin_dependent
    delta = np.asarray(b.gauge_origin) - np.asarray(a.gauge_origin)
    diag_a = np.real(np.diagonal(a.d, axis1=1, axis2=2))
    diag_b = np.real(np.diagonal(b.d, axis1=1, axis2=2))
    assert np.abs(diag_b - (diag_a - 1.0 * delta[:, None])).max() < 1e-7
    # ... and the transition elements, compared through the phase-invariant line strength.
    sa = block_line_strengths(a.d, [(0, 1), (1, 1)])
    sb = block_line_strengths(b.d, [(0, 1), (1, 1)])
    assert sa[0, 1] == pytest.approx(sb[0, 1], rel=1e-6)


def test_the_inactive_share_is_computed_used_and_not_warned_about(lih_matrices, kuiva_caplog):
    """⚠ ``r`` is time **even**, so the inactive electrons carry a real share of the dipole and
    a warning about it would be a warning that the core exists. ``L`` and ``S`` are time odd and
    keep their warning; this is the ``expect_zero=False`` path and it must stay separate."""
    assert np.abs(lih_matrices.inactive_d).max() > 1e-3, "Li 1s^2 must carry dipole"
    kuiva_caplog.clear()
    dump.inactive_moment(np.zeros((3, 2, 2)) + 1.0, [0, 1], name="d", expect_zero=False)
    assert not [r for r in kuiva_caplog.records if r.levelname == "WARNING"]
    dump.inactive_moment(np.zeros((3, 2, 2)) + 1.0, [0, 1], name="L")
    assert any("time odd" in r.message for r in kuiva_caplog.records)


# --- the invariants ----------------------------------------------------------------------------

def test_the_reductions_survive_phases_and_block_rotations(lih_matrices):
    """⚠ The reason validation may go through these and nothing else.

    Degenerate states mix arbitrarily and every state carries an arbitrary phase, so an
    element-by-element comparison of ``d`` compares neither. ``Tr_b(d_i d_j)`` and the
    block-to-block line strength are invariant under both, and here they are put through a
    random one of each.
    """
    d = np.asarray(lih_matrices.d)
    blocks = [(m.start, m.size) for m in lih_matrices.analyse()]
    rng = np.random.default_rng(11)
    n = d.shape[-1]
    u = np.eye(n, dtype=complex)
    for start, size in blocks:
        block = rng.standard_normal((size, size)) + 1j * rng.standard_normal((size, size))
        q, _ = np.linalg.qr(block)
        u[start:start + size, start:start + size] = q
    u = u * np.exp(2j * np.pi * rng.random(n))[None, :]
    mixed = np.stack([u.conj().T @ dk @ u for dk in d])

    assert np.abs(mixed - d).max() > 1e-3, "the scrambling must actually change the matrices"
    for start, size in blocks:
        assert block_dipole_tensor(mixed, start, size) == pytest.approx(
            block_dipole_tensor(d, start, size), abs=1e-10)
    assert block_line_strengths(mixed, blocks) == pytest.approx(
        block_line_strengths(d, blocks), abs=1e-10)


def test_the_spin_selection_rule_is_exact(lih_matrices):
    """With spin-orbit coupling off the dipole is spin free, so singlet -> triplet is forbidden.

    An analytic target that needs no reference: the LiH spectrum here is a singlet ground state,
    a triplet, and a singlet, and the line strength to the **triplet block** must vanish while
    the one to the excited singlet must not.
    """
    blocks = lih_matrices.analyse()
    assert [m.size for m in blocks[:3]] == [1, 3, 1], [m.size for m in blocks]
    s = lih_matrices.line_strengths(multiplets=blocks)
    assert s[0, 1] < 1e-18
    assert s[0, 2] > 1e-2


def test_transition_dipole_matches_an_independent_pyscf_pipeline(lih_matrices):
    """⚠ **The check whose two sides share no implementation above libcint.**

    A whole PySCF pipeline of its own — ``sfx2c1e`` RHF, ``mcscf.CASCI``,
    ``fci.direct_spin1.trans_rdm1``, its own dipole integrals, its own core and nuclear terms —
    against Kuiva's two-component route. An in-house check cannot see an error in the assembly
    it shares; this one can, and it is what the whole file is graded against.

    Compared through the **line strength** rather than element by element, since the excited
    singlet is what the two codes both resolve and phases are arbitrary.
    """
    from pyscf import fci, gto, mcscf, scf

    mol = gto.M(atom="Li 0 0 0; H 0 0 1.6", basis=BASIS, verbose=0, unit="Angstrom")
    mf = scf.RHF(mol).sfx2c1e().run()
    mc = mcscf.CASCI(mf, 2, 2)
    mc.fcisolver.nroots = 3
    mc.kernel()

    weights = mol.atom_mass_list()
    com = mol.atom_coords().T @ weights / weights.sum()
    mol.set_common_orig(com)
    r_ao = mol.intor("int1e_r", comp=3)
    core, cas = mc.mo_coeff[:, :mc.ncore], mc.mo_coeff[:, mc.ncore:mc.ncore + mc.ncas]
    r_cas = np.einsum("pi,kpq,qj->kij", cas, r_ao, cas)
    core_r = 2.0 * np.einsum("pi,kpq,qi->k", core, r_ao, core)
    z = np.asarray([mol.atom_charge(i) for i in range(mol.natm)])
    nuc = (mol.atom_coords() - com).T @ z

    def dipole(i, j):
        rdm = fci.direct_spin1.trans_rdm1(mc.ci[i], mc.ci[j], mc.ncas, 2)
        electronic = -np.einsum("kij,ji->k", r_cas, rdm)
        if i == j:
            return electronic - core_r + nuc
        return electronic

    blocks = lih_matrices.analyse()
    # State 0 is non-degenerate, so its dipole moment is a number both codes name unambiguously.
    ours = np.real(np.asarray(lih_matrices.d)[:, 0, 0])
    assert ours == pytest.approx(dipole(0, 0), abs=1e-5)
    assert abs(ours[2]) > 1.0, "LiH must have a substantial dipole; a zero would prove nothing"

    # The transition to the excited singlet, through the invariant.
    theirs = float(np.sum(np.abs(dipole(0, 2)) ** 2))
    ours_s = lih_matrices.line_strengths(multiplets=blocks)[0, 2]
    assert ours_s == pytest.approx(theirs, rel=1e-4)
    assert theirs > 1e-2


# --- the file ----------------------------------------------------------------------------------

def test_the_file_carries_the_dipole_its_units_and_its_nuclear_half(lih_matrices, tmp_path):
    """Round-tripped through the parser, not inspected as strings — a format nobody reads back
    has an undetected ambiguity in it. ⚠ ``format_version`` must **not** move: the rule is that it
    tracks a change in the *meaning* of a stored field, not the arrival of a new one, so an
    existing consumer of ``H`` and ``mu`` is unaffected."""
    path = lih_matrices.write(tmp_path / "lih.props")
    back = dump.read_dump(path)
    assert int(back["header"]["format_version"]) == dump.FORMAT_VERSION
    assert back["header"]["dipole_unit"] == "e*a0"
    assert back["header"]["dipole_includes_nuclear"].startswith("yes")
    assert back["header"]["molecular_charge"] == "0"
    assert back["header"]["dipole_origin_dependence"].startswith("none")
    assert back["header"]["picture_change_on_dipole"] == "none"
    nuclear = np.array([float(x) for x in back["header"]["nuclear_dipole_ea0"].split()])
    assert nuclear == pytest.approx(lih_matrices.nuclear_dipole, abs=1e-10)
    for k, axis in enumerate("xyz"):
        assert np.array_equal(back["matrices"]["d_" + axis], lih_matrices.d[k])
    # The [INACTIVE] rows are written at the same six-digit precision L and S have used since
    # the format existed; a tighter tolerance here would be testing the format, not the value.
    assert back["inactive"]["d"] == pytest.approx(lih_matrices.inactive_d, rel=1e-6)

    restored = dump.PropertyMatrices.from_dump(path)
    assert restored.has_dipole
    assert np.array_equal(restored.d, lih_matrices.d)
    assert restored.molecular_charge == 0
    assert restored.line_strengths() == pytest.approx(lih_matrices.line_strengths(), abs=1e-12)


def test_a_file_without_a_dipole_reads_back_as_none_not_as_zeros(lih_matrices, tmp_path):
    """⚠ "no dipole was computed" and "the dipole is zero" are different statements, and a
    symmetric molecule makes the second one true. Zero-filling would merge them."""
    path = lih_matrices.write(tmp_path / "no_d.props", include_dipole=False)
    back = dump.read_dump(path)
    assert not any(name.startswith("d_") for name in back["matrices"])
    assert "dipole_unit" not in back["header"]
    restored = dump.PropertyMatrices.from_dump(path)
    assert restored.d is None and not restored.has_dipole
    with pytest.raises(ValueError, match="no electric dipole"):
        restored.line_strengths()


def test_a_charged_system_says_so_in_the_warning_and_in_the_file(tmp_path, kuiva_caplog):
    """The file outlives the session, so the origin dependence is stated in both places."""
    ref, result = _lih(charge=1, spin=1, n_states=2)
    matrices = api.property_matrices(ref, result)
    kuiva_caplog.clear()
    path = matrices.write(tmp_path / "cation.props")
    messages = [r.message for r in kuiva_caplog.records if r.levelname == "WARNING"]
    assert any("origin dependent" in m and "Transition elements" in m for m in messages), messages
    text = path.read_text()
    assert "molecular_charge                 1" in text
    assert "CHARGED" in text
    assert dump.read_dump(path)["header"]["dipole_origin_dependence"].startswith("diagonal")


def test_the_report_prints_only_invariants(lih_matrices, kuiva_caplog):
    """⚠ A single ``d[I,J]`` in the log would be a phase. What is printed is the block-averaged
    permanent moment and the line strength, both invariant — and the line says the strength is
    not an oscillator strength, because that is the reading a user will reach for."""
    kuiva_caplog.clear()
    lih_matrices.report()
    text = "\n".join(r.message for r in kuiva_caplog.records)
    assert "Electric dipole" in text
    assert "S to block 0" in text
    assert "NOT an oscillator strength" in text
