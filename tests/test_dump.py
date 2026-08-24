"""Tests for the property operators and the property-matrix dump.

This is the file the whole program produces, so the tests are graded by what they can *fail
on*, not by how many digits they agree to:

**Tier 0, exact and analytic.** ``S`` closes the SU(2) algebra exactly in any orthonormal
spinor basis; ``L^2`` is exactly ``2 hbar^2`` on an atomic p shell, since the three functions
of one shell span ``l = 1`` exactly; both ``L`` and ``S`` are time odd, so a Kramers pair
contributes exactly zero. None of these involve a comparison with another program.

**The decisive one is the relative sign.** A sign error between ``L`` and ``S`` leaves every
operator Hermitian, every norm right and every degeneracy intact, and it changes every g value
in the file — it is exactly the sign-convention failure the spin-orbit integrals warn about
themselves. It is pinned by :func:`test_p_shell_j_quantum_numbers`: restricted to one atomic p
shell the X2C Hamiltonian *is* ``zeta L.S``, so its eigenvectors must carry
``<J^2> = j(j+1)`` with the ``j = 1/2`` doublet **below** the ``j = 3/2`` quartet. That fixes
the sign of ``L`` against ``S`` against the spin-orbit operator, all three at once, with no
SCF and no reference data.

**End to end against angular-momentum theory.** A state-averaged CASSCF on the ``2p^1`` boron
atom must give the ``(2, 4)`` pattern with ``g = 2/3`` and ``g = 4/3`` — the same class of
analytic target that is the sharpest check in the suite, and reachable in about a second.

**The file.** Round-tripped through :func:`kuiva.props.dump.read_dump` rather than inspected as
strings; a format nobody reads back has an undetected ambiguity in it.

⚠ **Every comparison of moment matrices here goes through
:meth:`~kuiva.props.dump.PropertyMatrices.analyse`**, — including the test that says
why (:func:`test_analysis_survives_arbitrary_phases_and_block_rotations`).
"""
import json

import numpy as np
import pytest

from kuiva.interface import api
from kuiva.interface.pyscf_bridge import (ao_layout, gauge_origin_for, ingest_property_integrals,
                                          ingest_spin_orbit)
from kuiva.mcscf.orbopt import OrbitalSpaces
from kuiva.props import dump
from kuiva.props.dump import FORMAT_VERSION, PropertyMatrices
from kuiva.props.multiplet import G_ELECTRON, lande_g
from kuiva.spinor.expand import expand_scalar_mos, spin_operator, time_reverse

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# --- fixtures -------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def argon():
    """A closed-shell atom with an unambiguous 3p spin-orbit splitting. No SCF needed."""
    from pyscf import gto
    return gto.M(atom="Ar 0 0 0", basis="x2c-SVPall-2c", verbose=0)


@pytest.fixture(scope="module")
def argon_operators(argon):
    """``(H_2c, L, S, C)`` over the spinors of the **outermost p shell** of the AO basis.

    One shell of real solid harmonics spans ``l = 1`` exactly and its three functions are
    mutually orthogonal and equally normalized, so no SCF and no orthogonalization is needed:
    normalizing the three columns already gives an orthonormal scalar basis for the shell,
    and the Kramers expansion turns it into six orthonormal spinors.
    """
    soc = ingest_spin_orbit(argon, screening="none")
    props = ingest_property_integrals(argon)
    s = argon.intor("int1e_ovlp")
    layout = ao_layout(argon)
    p_ao = np.nonzero((np.asarray(layout.ao_atom) == 0) &
                      (np.asarray(layout.ao_l) == 1))[0]
    idx = p_ao[-3:]                                   # the most diffuse p shell
    c = np.zeros((argon.nao, 3))
    c[idx, np.arange(3)] = 1.0
    c /= np.sqrt(np.diag(c.T @ s @ c))
    assert np.allclose(c.T @ s @ c, np.eye(3), atol=1e-13), "one p shell is not orthonormal"

    spinors = expand_scalar_mos(c, basis="ao").c
    ct = spinors.conj().T
    h = ct @ soc.hamiltonian() @ spinors
    l = np.stack([ct @ lk @ spinors for lk in props.two_component()])
    sp = np.stack([ct @ sk @ spinors for sk in spin_operator(s)])
    return h, l, sp, spinors


@pytest.fixture(scope="module")
def boron():
    """A state-averaged CASSCF on ``B 2p^1`` — the analytic ``g = 2/3, 4/3`` case.

    Six equally weighted roots restore the spherical symmetry ROHF breaks by putting the odd
    electron in one 2p orbital, which is what makes the free-ion g factors analytic here at
    all: on the ROHF orbitals the shell is split by ~5000 cm^-1 and the moments are quenched.
    """
    mol = api.Molecule(atoms=[("B", (0.0, 0.0, 0.0))], basis="x2c-SVPall-2c", spin=1)
    reference = api.spinor_reference(mol, screening="none", memory_gb=8)
    outcome = api.casscf(reference, character=("B", "p"), n_active=6, n_active_elec=1,
                         n_states=6, mode="second-order", conv_grad=1e-6, report=False)
    assert outcome.converged
    return reference, outcome


@pytest.fixture(scope="module")
def boron_matrices(boron):
    reference, outcome = boron
    return api.property_matrices(reference, outcome)


# --- Tier 0: the operators themselves --------------------------------------------------------

def test_spin_closes_the_su2_algebra_exactly(argon_operators):
    """``[S_x, S_y] = i S_z`` and ``S^2 = 3/4``, to machine precision.

    Exact rather than approximate, and that is the point: spin space is two dimensional and
    complete, so unlike ``L`` these identities do not depend on the basis being able to
    represent the operator's action. A factor of two anywhere in :func:`spin_operator` breaks
    both.
    """
    _, _, s, _ = argon_operators
    n = s.shape[1]
    comm = s[0] @ s[1] - s[1] @ s[0]
    assert np.abs(comm - 1j * s[2]).max() < 1e-13
    s2 = sum(sk @ sk for sk in s)
    assert np.abs(s2 - 0.75 * np.eye(n)).max() < 1e-13


def test_operators_are_hermitian(argon_operators):
    _, l, s, _ = argon_operators
    for name, op in (("L", l), ("S", s)):
        assert np.abs(op - op.conj().transpose(0, 2, 1)).max() < 1e-13, name


def test_angular_momentum_squared_on_an_atomic_p_shell(argon_operators):
    """``L^2 = 2 hbar^2`` on one p shell — ``l(l+1)`` with ``l = 1``, exactly.

    The three real solid harmonics of a single shell span the ``l = 1`` irrep exactly, so this
    holds with no basis-set error at all. It fixes the **normalization** of ``L``: PySCF's
    ``int1e_cg_irxp`` is ``<mu| r x nabla |nu>`` and the physical operator carries a factor
    ``-i``, so a missing or doubled factor shows up here immediately.
    """
    _, l, _, _ = argon_operators
    l2 = sum(lk @ lk for lk in l)
    assert np.abs(l2 - 2.0 * np.eye(l.shape[1])).max() < 1e-11


def test_l_and_s_are_time_odd_over_a_kramers_pair(argon):
    """``<psi|A|psi> + <T psi|A|T psi> = 0`` for both ``L`` and ``S``.

    This is the theorem the property dump relies on to drop the inactive contribution — except that
    it does not drop it, it computes it (:func:`kuiva.props.dump.inactive_moment`). The test is
    on the *orbitals*, and it is what makes a Kramers-paired inactive space carry no moment.
    """
    props = ingest_property_integrals(argon)
    s_ao = argon.intor("int1e_ovlp")
    rng = np.random.default_rng(7)
    c = np.linalg.qr(rng.standard_normal((argon.nao, 4)))[0]
    spinors = expand_scalar_mos(c, basis="ao").c
    assert np.abs(spinors[:, 1::2] - time_reverse(spinors[:, 0::2])).max() < 1e-13

    for op in (props.two_component(), spin_operator(s_ao)):
        for k in range(3):
            diag = np.real(np.diag(spinors.conj().T @ op[k] @ spinors))
            pairs = diag.reshape(-1, 2).sum(axis=1)
            assert np.abs(pairs).max() < 1e-12


def test_p_shell_j_quantum_numbers(argon_operators):
    """⚠ **The decisive test: the relative sign of L, S and the spin-orbit operator.**

    Restricted to one atomic p shell the two-component X2C Hamiltonian is spherically
    symmetric, hence ``a + zeta L.S`` exactly, so its eigenvectors are ``j`` eigenstates. With
    ``J = L + S`` built from *this module's* operators, the six states must split **2 + 4**
    with ``<J^2> = 3/4`` on the lower pair and ``15/4`` on the upper four — Kramers' theorem
    plus Hund's third rule for a less-than-half-filled shell.

    Flip the sign of ``L`` and every operator stays Hermitian, every degeneracy survives, the
    splitting keeps its magnitude, and this test fails — which is exactly the failure mode
    recorded for the spin-orbit integrals and for the multiplet ordering.
    """
    h, l, s, _ = argon_operators
    j = l + s
    j2 = sum(jk @ jk for jk in j)
    energies, vectors = np.linalg.eigh(h)
    expectation = np.real(np.diag(vectors.conj().T @ j2 @ vectors))

    splitting = energies[2] - energies[0]
    assert splitting > 1e-6, "no spin-orbit splitting to order the multiplets by"
    assert abs(energies[1] - energies[0]) < 1e-10 * abs(splitting)      # Kramers, exact
    # The quartet is degenerate by spherical symmetry rather than by time reversal, so it
    # inherits the rounding of a matrix whose diagonal is ~9 Eh: measured at 4e-12 Eh, six
    # orders below the splitting it has to be distinguished from.
    assert np.abs(energies[2:] - energies[2]).max() < 1e-6 * abs(splitting)

    assert expectation[:2] == pytest.approx(0.75, abs=1e-9), "the lower block is not j = 1/2"
    assert expectation[2:] == pytest.approx(3.75, abs=1e-9), "the upper block is not j = 3/2"


# --- the gauge origin ------------------------------------------------------------------------

def test_gauge_origin_choices(argon):
    for spec, label in ((None, "centre of mass"), ("mass", "centre of mass"),
                        ("charge", "centre of nuclear charge"),
                        ("origin", "coordinate origin")):
        r, got = gauge_origin_for(argon, spec)
        assert got == label and np.asarray(r).shape == (3,)
    r, label = gauge_origin_for(argon, (1.0, 2.0, 3.0))
    # ⚠ The bare tuple keeps its bohr meaning -- no stored number moves -- but the label now
    # records that the unit was *assumed* rather than declared, which is what a reader of the
    # header needs in order to know how much to trust it.
    assert label == "explicit (bohr, untagged)" and np.allclose(r, [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="unknown gauge origin"):
        gauge_origin_for(argon, "centroid")
    with pytest.raises(ValueError, match="three coordinates"):
        gauge_origin_for(argon, (1.0, 2.0))


def test_a_bare_gauge_origin_tuple_warns_about_its_unit(argon, kuiva_caplog):
    """⚠ **The trap, and why a warning on correct usage is worth it.**

    The tuple has always meant bohr and still does, so nothing breaks and no stored number
    moves. But ``Molecule`` geometry is in *Angstrom*, so a coordinate copied from the
    geometry into ``gauge_origin=`` lands 1.89x too far out — with no error, no exception and
    nothing in the output that looks wrong. It shifts the point ``L`` is defined about, so
    every orbital moment in the dump is wrong and every one of them is a plausible number.
    """
    gauge_origin_for(argon, (1.0, 2.0, 3.0))
    assert any("read as BOHR" in r.message and "('angstrom'" in r.message
               for r in kuiva_caplog.records)

    kuiva_caplog.clear()
    r_b, label_b = gauge_origin_for(argon, ("bohr", 1.0, 2.0, 3.0))
    r_a, label_a = gauge_origin_for(argon, ("angstrom", 1.0, 2.0, 3.0))
    assert not kuiva_caplog.records                      # saying which unit silences it
    assert np.allclose(r_b, [1.0, 2.0, 3.0])             # ...and bohr means what it did
    assert np.allclose(r_a, np.array([1.0, 2.0, 3.0]) / 0.52917721092, rtol=1e-6)
    assert label_b == "explicit (bohr)" and label_a == "explicit (angstrom)"


def test_the_gauge_origin_can_sit_on_an_atom():
    """``("atom", k)`` through the same addressing bases and configurations use, so there is
    one way to name an atom in this program rather than three."""
    from pyscf import gto

    mol = gto.M(atom="Ti 0 0 0; Cl 2.25 0 0; Cl -1.125 1.949 0; Cl -1.125 -1.949 0",
                basis="sto-3g", spin=1, verbose=0)
    coords = mol.atom_coords()
    for key in (1, "Ti", "Ti1"):
        r, label = gauge_origin_for(mol, ("atom", key))
        assert np.allclose(r, coords[0]) and label == "atom 1 (Ti)"
    r, label = gauge_origin_for(mol, ("atom", "Cl3"))
    assert np.allclose(r, coords[2]) and label == "atom 3 (Cl)"

    # ⚠ An element naming several atoms names no *point*, and is refused rather than
    # resolved to the first -- which no output would have revealed.
    with pytest.raises(ValueError, match="names 3 atoms"):
        gauge_origin_for(mol, ("atom", "Cl"))
    with pytest.raises(ValueError, match="out of range"):
        gauge_origin_for(mol, ("atom", 9))
    with pytest.raises(ValueError, match="unknown gauge-origin form"):
        gauge_origin_for(mol, ("centroid", 1, 2, 3))


def test_the_atom_form_reads_pure_element_symbols_not_decorated_ones():
    """⚠ A molecule with a per-atom basis or reference state carries PySCF *decorated*
    symbols (``"Ti2"``), and the label ``"Ti2"`` means "atom 2, which is a Ti" everywhere
    else in this program. Handing the decorated symbols to the shared parser would compare a
    label against a label and answer a different question — for exactly the molecules the
    labels exist for."""
    from pyscf import gto

    mol = gto.M(atom="Ti2 0 0 0; O 0 0 2.0", basis={"Ti2": "sto-3g", "O": "sto-3g"},
                spin=2, verbose=0)
    assert mol.atom_symbol(0) == "Ti2"                   # decorated, and not what we address by
    r, label = gauge_origin_for(mol, ("atom", "Ti"))
    assert np.allclose(r, mol.atom_coords()[0]) and label == "atom 1 (Ti)"
    r, label = gauge_origin_for(mol, ("atom", "O2"))     # atom 2, which is an O
    assert np.allclose(r, mol.atom_coords()[1]) and label == "atom 2 (O)"


def test_gauge_origin_changes_the_angular_momentum():
    """``L`` is defined relative to the gauge origin, so moving it moves ``L`` — which is why
    the origin is recorded in the dump header rather than left implicit."""
    from pyscf import gto
    mol = gto.M(atom="Ne 0 0 1.5", basis="sto-3g", unit="Bohr", verbose=0)
    at_com = ingest_property_integrals(mol, "mass")
    at_zero = ingest_property_integrals(mol, "origin")
    assert not np.allclose(at_com.gauge_origin, at_zero.gauge_origin)
    assert np.abs(at_com.irxp - at_zero.irxp).max() > 1e-6
    # About the atom itself the two agree, since the atom *is* the centre of mass there.
    centred = ingest_property_integrals(gto.M(atom="Ne 0 0 0", basis="sto-3g", verbose=0),
                                        "origin")
    assert np.abs(at_com.irxp - centred.irxp).max() < 1e-12


def test_property_integrals_provenance_is_json_serializable(argon):
    props = ingest_property_integrals(argon)
    text = json.dumps(props.provenance(), sort_keys=True)
    assert "gauge_origin_choice" in text and "picture_change" in text


# --- lifting into the state basis -------------------------------------------------------------

def test_state_operator_matrices_against_a_direct_contraction():
    """``A^{IJ} = sum_tu A_tu gamma^{IJ}_tu``, against the expression written out.

    An independent contraction rather than the same reshape twice — the GEMM in
    :func:`kuiva.props.dump.state_operator_matrices` flattens the orbital pair index, and a
    transposed flattening is the plausible-and-wrong failure here (the ``F`` index-order
    trap in miniature).
    """
    rng = np.random.default_rng(11)
    na, ns = 4, 3
    op = rng.standard_normal((3, na, na)) + 1j * rng.standard_normal((3, na, na))
    tdm = rng.standard_normal((ns, ns, na, na)) + 1j * rng.standard_normal((ns, ns, na, na))
    got = dump.state_operator_matrices(op, tdm)
    want = np.einsum("ktu,IJtu->kIJ", op, tdm)
    assert np.abs(got - want).max() < 1e-13

    trace = np.array([1.0, -2.0, 0.5])
    shifted = dump.state_operator_matrices(op, tdm, trace)
    assert np.abs(shifted - got - trace[:, None, None] * np.eye(ns)).max() < 1e-13


def test_state_operator_matrices_reject_mismatched_shapes():
    tdm = np.zeros((2, 2, 3, 3), dtype=complex)
    with pytest.raises(ValueError, match=r"\(3, n_act, n_act\)"):
        dump.state_operator_matrices(np.zeros((2, 3, 3)), tdm)
    with pytest.raises(ValueError, match="transition densities"):
        dump.state_operator_matrices(np.zeros((3, 4, 4)), tdm)


def test_inactive_moment_vanishes_for_a_kramers_paired_space(argon):
    """The theorem, applied where the dump applies it: a Kramers-paired inactive space
    contributes exactly nothing to ``L`` or ``S``."""
    props = ingest_property_integrals(argon)
    s_ao = argon.intor("int1e_ovlp")
    rng = np.random.default_rng(3)
    c = np.linalg.qr(rng.standard_normal((argon.nao, 5)))[0]
    spinors = expand_scalar_mos(c, basis="ao").c
    l_mo, s_mo = dump.spinor_operators(spinors, props.two_component(), spin_operator(s_ao))
    for op, name in ((l_mo, "L"), (s_mo, "S")):
        got = dump.inactive_moment(op, np.arange(6), name=name)
        assert np.abs(got).max() < 1e-12


def test_inactive_moment_warns_when_the_space_is_not_kramers_paired(argon, kuiva_caplog):
    """Half a Kramers pair carries a moment, and the user is told rather than the term being
    dropped: a moment matrix silently missing a core contribution looks entirely plausible."""
    props = ingest_property_integrals(argon)
    s_ao = argon.intor("int1e_ovlp")
    rng = np.random.default_rng(3)
    c = np.linalg.qr(rng.standard_normal((argon.nao, 5)))[0]
    spinors = expand_scalar_mos(c, basis="ao").c
    _, s_mo = dump.spinor_operators(spinors, props.two_component(), spin_operator(s_ao))
    got = dump.inactive_moment(s_mo, [0, 1, 2], name="S")            # splits a pair
    assert np.abs(got).max() > 1e-3
    assert any("no longer Kramers paired" in r.message for r in kuiva_caplog.records)


def test_inactive_moment_of_an_empty_space_is_zero():
    assert np.array_equal(dump.inactive_moment(np.zeros((3, 4, 4)), []), np.zeros(3))


# --- end to end: the analytic free-ion g factors ------------------------------------------------

def test_boron_multiplet_structure(boron_matrices):
    """``2p^1`` gives ``2P_1/2`` below ``2P_3/2``: the (2, 4) pattern, exactly degenerate."""
    blocks = boron_matrices.analyse()
    assert [b.size for b in blocks] == [2, 4]
    assert blocks[0].energy_cm == pytest.approx(0.0, abs=1e-6)
    assert blocks[1].energy_cm > 1.0
    for b in blocks:
        assert b.spread_cm < 1e-3


def test_boron_lande_g_values_are_analytic(boron_matrices):
    """``g = 2/3`` (j = 1/2) and ``g = 4/3`` (j = 3/2), to 1%.

    ⚠ The target is angular-momentum theory, not another program — which makes this the
    sharpest check available of the whole chain: spin-orbit operator, CI states, transition
    densities, ``L`` and ``S``, and the phase-invariant reduction. The residual 0.12%
    deviation is real physics, ``g_e - 2`` (the analytic value uses ``g_e = 2``), and it has
    the right sign in both blocks.
    """
    blocks = boron_matrices.analyse()
    for block, (l, s) in zip(blocks, ((1, 0.5), (1, 0.5))):
        expected = lande_g(l, s, block.j)
        for g in block.g_values:
            assert g == pytest.approx(expected, rel=0.01), \
                "j = {}: got {}, expected {:.6f}".format(block.j, block.g_values, expected)
        # The free-electron g factor shifts g away from the g_e = 2 value, up for j = 3/2 and
        # down for j = 1/2, by (g_e - 2)/3 in both cases.
        shift = (G_ELECTRON - 2.0) / 3.0
        target = expected + (shift if block.j > 1 else -shift)
        assert block.g_iso == pytest.approx(target, abs=1e-4)


def test_boron_g_tensor_is_isotropic(boron_matrices):
    """A free atom has no preferred direction; anisotropy would mean a spurious axis."""
    for block in boron_matrices.analyse():
        assert max(block.g_values) - min(block.g_values) < 1e-4


def test_moment_matrices_are_hermitian(boron_matrices):
    assert boron_matrices.hermiticity_error() < 1e-12


def test_inactive_space_carries_no_moment_in_a_real_calculation(boron_matrices):
    assert np.abs(boron_matrices.inactive_l).max() < dump.DEFAULT_INACTIVE_TOL
    assert np.abs(boron_matrices.inactive_s).max() < dump.DEFAULT_INACTIVE_TOL


def test_analysis_survives_arbitrary_phases_and_block_rotations(boron_matrices):
    """⚠ **Why every comparison here goes through** :meth:`PropertyMatrices.analyse`.

    the dump fixes no phase convention and degenerate states mix arbitrarily, so ``mu`` itself is
    not reproducible. Applying random phases *and* a random unitary mixing inside each
    degenerate block changes every matrix element and must change no invariant.
    """
    rng = np.random.default_rng(19)
    blocks = boron_matrices.analyse()
    n = boron_matrices.n_states
    u = np.zeros((n, n), dtype=complex)
    for b in blocks:
        sl = slice(b.start, b.start + b.size)
        q = np.linalg.qr(rng.standard_normal((b.size, b.size)) +
                         1j * rng.standard_normal((b.size, b.size)))[0]
        u[sl, sl] = q
    phases = np.exp(2j * np.pi * rng.random(n))
    u = u * phases[None, :]

    rotated = np.stack([u.conj().T @ mk @ u for mk in boron_matrices.mu])
    assert np.abs(rotated - boron_matrices.mu).max() > 1e-3, "the rotation did nothing"

    mixed = dump.PropertyMatrices(energies=boron_matrices.energies, mu=rotated,
                                  l=boron_matrices.l, s=boron_matrices.s)
    for a, b in zip(boron_matrices.analyse(), mixed.analyse()):
        assert a.size == b.size
        assert a.energy_cm == pytest.approx(b.energy_cm, abs=1e-8)
        assert np.allclose(sorted(a.g_values), sorted(b.g_values), atol=1e-10)


def test_property_matrices_refuse_a_result_that_lost_its_orbitals(boron):
    """A moment matrix built from one orbital set and states solved at another is Hermitian,
    plausible and wrong, so the pairing is required rather than assumed."""
    from kuiva.mcscf.casci import CASCIResult
    reference, outcome = boron
    orphan = CASCIResult(energies=outcome.ci.energies, vectors=outcome.ci.vectors,
                         weights=outcome.ci.weights, gamma=outcome.ci.gamma,
                         gamma2=outcome.ci.gamma2, e_core=outcome.ci.e_core,
                         n_apply=0, n_iter=0)
    with pytest.raises(ValueError, match="which orbitals"):
        api.property_matrices(reference, orphan)


def test_casci_result_records_its_orbitals(boron):
    """Both drivers record the orbitals and the partition the states belong to, which is what
    makes the dump reachable from either a CASCI or a CASSCF."""
    reference, outcome = boron
    assert outcome.ci.coeff is not None and outcome.ci.spaces is not None
    assert np.allclose(outcome.ci.coeff, outcome.coeff)
    result = api.casci(reference, character=("B", "p"), n_active=6, n_active_elec=1,
                       n_states=6, coeff=outcome.coeff, report=False)
    assert result.coeff is not None and result.spaces is not None
    assert result.description
    # A CASCI on the converged CASSCF orbitals is the same calculation (the 1e-7 band).
    assert result.total_energies == pytest.approx(outcome.ci.total_energies, abs=1e-7)


# --- the file ----------------------------------------------------------------------------------

def test_dump_round_trip_is_exact(boron_matrices, tmp_path):
    """Written and read back element for element. The parser is part of the contract: a format
    nobody reads back has an undetected ambiguity in it."""
    path = boron_matrices.write(tmp_path / "b.prop", title="round trip")
    back = dump.read_dump(path)
    assert back["energies"] == pytest.approx(boron_matrices.energies, abs=0.0, rel=0.0)
    expected = {"H": boron_matrices.hamiltonian}
    for k, axis in enumerate("xyz"):
        expected["mu_" + axis] = boron_matrices.mu[k]
        expected["L_" + axis] = boron_matrices.l[k]
        expected["S_" + axis] = boron_matrices.s[k]
    assert set(back["matrices"]) == set(expected)
    for name, want in expected.items():
        assert np.array_equal(back["matrices"][name], want), name
    assert np.allclose(back["inactive"]["L"], boron_matrices.inactive_l, atol=1e-9)


def test_dump_hamiltonian_is_diagonal(boron_matrices, tmp_path):
    """⚠ Unlike RASSI's: this CI is already two-component, so its roots **are** the spin-orbit
    eigenstates. The header says so, and here it is."""
    back = dump.read_dump(boron_matrices.write(tmp_path / "b.prop"))
    h = back["matrices"]["H"]
    assert np.abs(h - np.diag(np.diag(h))).max() == 0.0
    assert back["header"]["hamiltonian_is_diagonal"] == "yes"
    assert np.diag(h).imag.max() == 0.0


def test_dump_header_carries_the_full_hamiltonian_provenance(boron_matrices, tmp_path):
    """The screening and decoupling records are contracts with stored data, and this file is
    where that obligation is discharged: a stored property matrix that does not say whether
    the two-electron picture change was included is not interpretable."""
    back = dump.read_dump(boron_matrices.write(tmp_path / "b.prop"))
    ham = back["provenance"]["hamiltonian"]
    assert ham["screening"]["method"] == "none"
    assert ham["decoupling"]["decoupling"] == "1e"
    assert ham["method"]
    assert back["header"]["gauge_origin_choice"] == "centre of mass"
    assert back["header"]["picture_change_on_properties"] == "none"
    assert back["header"]["active_space"] and back["header"]["active_space"] != "unspecified"
    assert back["provenance"]["n_active_electrons"] == 1


def test_dump_warns_about_the_missing_picture_change(boron_matrices, tmp_path, kuiva_caplog):
    """D2's standing obligation, announced every time the file is written and recorded in it.
    Not configurable: the file outlives the session that made it."""
    boron_matrices.write(tmp_path / "b.prop")
    assert any("NO picture-change" in r.message for r in kuiva_caplog.records)
    assert "picture-change transformation is applied" in (tmp_path / "b.prop").read_text()


def test_read_dump_refuses_an_unknown_format_version(boron_matrices, tmp_path):
    """The version exists so a consumer can refuse rather than misinterpret."""
    path = boron_matrices.write(tmp_path / "b.prop")
    text = path.read_text().replace("format_version                   {}".format(
        dump.FORMAT_VERSION), "format_version                   99")
    path.write_text(text)
    with pytest.raises(ValueError, match="refusing to guess"):
        dump.read_dump(path)


def test_dump_leaves_no_partial_file(boron_matrices, tmp_path):
    """Written whole, then moved into place: a truncated dump is worse than none, since it
    parses. Same discipline as the checkpoint writer."""
    boron_matrices.write(tmp_path / "b.prop")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["b.prop"]


def test_dump_can_omit_l_and_s(boron_matrices, tmp_path):
    back = dump.read_dump(boron_matrices.write(tmp_path / "b.prop", include_l_s=False))
    assert set(back["matrices"]) == {"H", "mu_x", "mu_y", "mu_z"}


def test_dump_threshold_drops_small_elements(boron_matrices, tmp_path):
    """A sparsity threshold is a size knob, not a change of meaning: the elements that survive
    are the ones that were there."""
    full = dump.read_dump(boron_matrices.write(tmp_path / "full.prop"))
    thin = dump.read_dump(boron_matrices.write(tmp_path / "thin.prop", threshold=1e-6))
    assert (tmp_path / "thin.prop").stat().st_size < (tmp_path / "full.prop").stat().st_size
    kept = np.abs(full["matrices"]["mu_z"]) >= 1e-6
    assert np.array_equal(thin["matrices"]["mu_z"][kept], full["matrices"]["mu_z"][kept])
    assert np.all(thin["matrices"]["mu_z"][~kept] == 0.0)


def test_dump_round_trips_at_the_largest_state_count_the_suite_reaches(tmp_path):
    """126 states — ``dy3p``'s ⁶H/⁶F/⁶P manifold, the biggest the validation suite produces.

    The format is exercised at that size *here*, on synthetic Hermitian matrices, rather than
    behind a 126-root CASSCF: what is untested at scale is the writer and the parser, not the
    physics, and the physics has its own tests. Catches a row count, a field width or an
    ``O(n²)`` line-assembly cost that only shows up when 4 x 126² elements are written.
    """
    rng = np.random.default_rng(5)
    n = 126
    mu = rng.standard_normal((3, n, n)) + 1j * rng.standard_normal((3, n, n))
    mu = mu + mu.conj().transpose(0, 2, 1)
    energies = np.sort(rng.standard_normal(n)) * 1e-3 - 1234.5
    matrices = dump.PropertyMatrices(energies=energies, mu=mu, l=mu, s=mu,
                                     active_space="synthetic 126-state manifold")
    path = matrices.write(tmp_path / "big.prop", include_l_s=False)
    back = dump.read_dump(path)
    assert back["header"]["n_states"] == str(n)
    assert np.array_equal(back["matrices"]["mu_y"], mu[1])
    assert back["energies"] == pytest.approx(energies, abs=0.0, rel=0.0)
    # 4 matrices x 126^2 elements, one line each, plus the header and energy blocks.
    assert path.read_text().count("\n") > 4 * n * n


def test_api_property_dump_reports_and_writes(boron, tmp_path, kuiva_caplog):
    reference, outcome = boron
    matrices = api.property_dump(reference, outcome, tmp_path / "b.prop",
                                 title="B 2p^1", report=True)
    assert (tmp_path / "b.prop").is_file()
    assert [b.size for b in matrices.analyse()] == [2, 4]


def test_spinor_operators_reject_a_mismatched_basis(argon):
    props = ingest_property_integrals(argon)
    s_ao = argon.intor("int1e_ovlp")
    bad = np.zeros((2 * argon.nao + 2, 4), dtype=complex)
    with pytest.raises(ValueError, match="two-component AO rows"):
        dump.spinor_operators(bad, props.two_component(), spin_operator(s_ao))
    with pytest.raises(ValueError, match=r"\(3, 2\*nao, 2\*nao\)"):
        dump.spinor_operators(np.zeros((2 * argon.nao, 4), dtype=complex),
                              props.two_component()[:2], spin_operator(s_ao))


def test_property_matrices_need_ingested_property_integrals(boron):
    """A reference built by hand — or by an older front-end — has no ``L``, and says so."""
    import dataclasses
    reference, outcome = boron
    stripped = dataclasses.replace(reference.data, properties=None)
    bare = type(reference)(data=stripped, orth=reference.orth, spinors=reference.spinors,
                           factors=reference.factors)
    with pytest.raises(ValueError, match="no property integrals"):
        api.property_matrices(bare, outcome)


def test_state_matrices_reproduce_the_orbital_expectation_values(boron):
    """The one-electron closure check: summed over a filled active space, the state moments
    must equal the orbital ones. ``sum_I <I|A|I>`` over a complete CI of ``N`` electrons in
    ``n`` spinors weights each orbital equally, so it is ``C(n-1, N-1)/C(n, N) * n`` times the
    orbital average — for the one-electron case here, simply the orbital trace."""
    reference, outcome = boron
    matrices = api.property_matrices(reference, outcome)
    from kuiva.spinor.expand import spin_operator as _spin
    l_mo, s_mo = dump.spinor_operators(outcome.coeff,
                                       reference.data.properties.two_component(),
                                       _spin(reference.data.s_ao))
    act = outcome.active.spaces.active
    for k in range(3):
        orbital = np.real(np.trace(l_mo[k][np.ix_(act, act)]))
        state = np.real(np.trace(matrices.l[k]))
        assert state == pytest.approx(orbital, abs=1e-10)


def test_orbital_spaces_partition_is_what_the_dump_uses(boron):
    reference, outcome = boron
    spaces = outcome.active.spaces
    assert isinstance(spaces, OrbitalSpaces)
    assert spaces.n_active == 6 and spaces.n_inactive == 4


# --- reading a stored dump back into a comparable object -----------------------------------

def test_a_dump_round_trips_through_from_dump(tmp_path):
    """⚠ **Why this exists at all.** The phases in this file are arbitrary, so the only sound
    way to compare two stored runs is through ``analyse()`` — and until ``from_dump`` existed
    that meant re-implementing the reduction against ``read_dump``'s dictionary at every call
    site that wanted it. The README mandates the comparison; this is what makes it one line.
    """
    rng = np.random.default_rng(11)
    n = 4

    def herm(k=3):
        a = rng.normal(size=(k, n, n)) + 1j * rng.normal(size=(k, n, n))
        return np.stack([0.5 * (m + m.conj().T) for m in a])

    original = PropertyMatrices(
        energies=np.array([-1.0, -1.0, -0.5, -0.5]), mu=herm(), l=herm(), s=herm(),
        gauge_origin=np.array([0.125, -0.25, 0.5]), origin_label="atom 1 (Ti)",
        active_space="5 lowest Kramers pairs of l=2 character on 1 Ti",
        provenance={"hamiltonian": {"method": "X2C-AMF"}}, comments=("a note",),
        inactive_l=np.array([1.5e-14, 0.0, 0.0]))
    path = original.write(tmp_path / "a.props", title="round trip")

    back = PropertyMatrices.from_dump(path)
    # Bitwise on the arrays: the writer prints enough digits, and a lossy round trip would
    # make the comparison this exists for a comparison of two different things.
    assert np.array_equal(back.mu, original.mu)
    assert np.array_equal(back.l, original.l) and np.array_equal(back.s, original.s)
    assert np.array_equal(back.energies, original.energies)
    assert np.allclose(back.gauge_origin, original.gauge_origin, atol=0, rtol=1e-12)
    assert back.origin_label == "atom 1 (Ti)"
    assert back.active_space == original.active_space
    assert back.provenance == original.provenance
    assert np.allclose(back.inactive_l, original.inactive_l)
    assert not back.picture_changed

    # ...and the reduction agrees, which is the actual deliverable.
    for a, b in zip(back.analyse(), original.analyse()):
        assert (a.size, a.g_values) == (b.size, b.g_values)
        assert a.g_sign == b.g_sign


def test_from_dump_survives_a_file_written_without_l_and_s(tmp_path):
    """``L`` and ``S`` are convenience, not the contract: ``mu`` is the quantity. A file
    written without them comes back usable, with the two zero-filled rather than invented."""
    rng = np.random.default_rng(12)
    n = 2
    mu = np.stack([(lambda a: 0.5 * (a + a.conj().T))(
        rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) for _ in range(3)])
    m = PropertyMatrices(energies=np.zeros(n), mu=mu, l=np.zeros((3, n, n), complex),
                         s=np.zeros((3, n, n), complex))
    path = m.write(tmp_path / "b.props", include_l_s=False)

    back = PropertyMatrices.from_dump(path)
    assert np.array_equal(back.mu, mu)
    assert not back.l.any() and not back.s.any()
    assert back.analyse()[0].g_values == pytest.approx(m.analyse()[0].g_values, rel=1e-12)


def test_from_dump_refuses_a_file_that_is_not_one(tmp_path):
    """A clear refusal beats an object full of zeros that analyses to nonsense."""
    path = tmp_path / "not.props"
    path.write_text("[HEADER]\nformat KUIVA_PROPERTY_MATRICES\nformat_version {}\n"
                    "n_states 2\n[END]\n[ENERGIES]\n0 -1.0\n1 -1.0\n[END]\n"
                    .format(FORMAT_VERSION))
    with pytest.raises(ValueError, match="not a property dump"):
        PropertyMatrices.from_dump(path)
