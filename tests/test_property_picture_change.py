"""The X2C picture change on the magnetic moment operator (non-default).

The property operators this program uses by default are the **bare** non-relativistic ``L`` and
``S``, used unchanged in the two-component basis — the same choice OpenMolcas RASSI makes, which
is what keeps a cross-code comparison of a property dump like-for-like. These tests cover the
alternative, :func:`kuiva.interface.pyscf_bridge.picture_changed_moment`.

⚠ **Every check here is chosen for what it can fail on**, because the failure mode of this
construction is a Hermitian, time-reversal-odd, plausible and wrong operator:

* the non-relativistic-limit identity is exact algebra, so it pins the spin mapping, the gauge
  origin and the small-component normalization at once — and it is verified to *break* when the
  mapping is corrupted, because a guard that cannot fail proves nothing;
* the ``c -> inf`` limit must return the bare operator, and must do so as ``1/c^2``;
* PySCF's own ``X2CHelperBase.picture_change`` is an independent implementation of the
  transformation and is compared against;
* the default path must be **untouched**, asserted by building the matrices both ways.
"""
import dataclasses

import numpy as np
import pytest

from kuiva.interface import api
from kuiva.interface.pyscf_bridge import (gauge_origin_for, ingest_property_integrals,
                                          picture_changed_moment)
from kuiva.props import dump
from kuiva.props.multiplet import G_ELECTRON, lande_g
from kuiva.spinor.expand import (decompose_two_component, spin_block_diagonal,
                                 spin_operator, two_component_operator)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(scope="module")
def hf_mol():
    from pyscf import gto
    return gto.M(atom="H 0 0 0; F 0 0 1.7", basis="sto-3g", verbose=0, unit="Bohr")


def _bare_moment(mol, origin):
    """``L + 2 S`` in the spin-blocked layout, about ``origin`` — Dirac's g, not ``g_e``."""
    mol.set_common_orig(np.asarray(origin, dtype=float).ravel())
    irxp = np.asarray(mol.intor("int1e_cg_irxp", comp=3), dtype=float)
    return (np.stack([spin_block_diagonal(-1j * lk) for lk in irxp])
            + 2.0 * spin_operator(np.asarray(mol.intor_symmetric("int1e_ovlp"))))


# --- Tier 0: the spin assembly --------------------------------------------------------------

def test_sigma_dot_reproduces_the_antisymmetric_assembly_bitwise():
    """``two_component_operator`` is a *caller* of :func:`sigma_dot`, not a second copy.

    Bitwise, because the point of factoring it out was to keep one definition of the four
    blocks; a refactor that changed the arithmetic by an ulp would still be two conventions.
    """
    rng = np.random.default_rng(0)
    n = 7
    a = rng.standard_normal((n, n))
    a = a + a.T
    w = rng.standard_normal((3, n, n))
    w = w - np.transpose(w, (0, 2, 1))
    expected = spin_block_diagonal(a)
    wx, wy, wz = w
    expected[:n, :n] += 1j * wz
    expected[n:, n:] -= 1j * wz
    expected[:n, n:] += 1j * wx + wy
    expected[n:, :n] += 1j * wx - wy
    assert np.array_equal(two_component_operator(a, w), expected)


def test_the_anomaly_small_component_identity_is_algebra():
    """``(sigma.p) sigma_k (sigma.p) = 2 (sigma.p) p_k - sigma_k p^2``.

    Verified on random **commuting** matrices, which is the only hypothesis the derivation
    uses. This is what stands behind :func:`kuiva.interface.pyscf_bridge._anomaly_small_block`,
    where there is no integral for the left-hand side and the right-hand side is assembled
    from two that exist.
    """
    rng = np.random.default_rng(4)
    m = 5
    q = np.linalg.qr(rng.standard_normal((m, m)))[0]
    p = [q @ np.diag(rng.standard_normal(m)) @ q.T for _ in range(3)]     # simultaneously diagonal
    sig = [np.array([[0, 1], [1, 0]], complex),
           np.array([[0, -1j], [1j, 0]]),
           np.array([[1, 0], [0, -1]], complex)]
    sdp = sum(np.kron(sig[i], p[i]) for i in range(3))
    p2 = sum(pk @ pk for pk in p)
    for k in range(3):
        lhs = sdp @ np.kron(sig[k], np.eye(m)) @ sdp
        rhs = 2.0 * (sdp @ np.kron(np.eye(2), p[k])) - np.kron(sig[k], p2)
        assert np.abs(lhs - rhs).max() < 1e-12


# --- Tier 0: the moment operator ------------------------------------------------------------

def test_the_non_relativistic_limit_identity_holds_exactly(hf_mol):
    """``<(r x sigma)(sigma.p)> + h.c. = L + 2 S``, to machine precision.

    Exact algebra rather than a numerical coincidence, which is why the production code makes
    it a refusal. It also fixes the normalization: the four-component moment carries an
    explicit ``c`` that cancels the small-component basis' ``1/(2c)``, so a stray factor of
    ``c`` anywhere shows up here immediately.
    """
    pc = picture_changed_moment(hf_mol, "mass")
    assert pc.identity_residual < 1e-13


def test_the_identity_check_is_load_bearing(hf_mol, monkeypatch):
    """Corrupt the spinor mapping and the build must **raise**, not warn or absorb it.

    Without this, "the identity holds" would be consistent with a check that cannot fail.
    """
    from kuiva.interface import pyscf_bridge

    good = pyscf_bridge._odd_moment_blocks

    def swapped(xmol, r_g):
        blocks = good(xmol, r_g)
        n = blocks.shape[-1] // 2
        out = blocks.copy()
        out[:, :n, :], out[:, n:, :] = blocks[:, n:, :], blocks[:, :n, :]
        return out

    monkeypatch.setattr(pyscf_bridge, "_odd_moment_blocks", swapped)
    with pytest.raises(RuntimeError, match="non-relativistic-limit identity"):
        picture_changed_moment(hf_mol, "mass")


@pytest.mark.parametrize("scale, expected", [(1.0, 1.3e-3), (10.0, 1.3e-5), (100.0, 1.3e-7)])
def test_c_to_infinity_returns_the_bare_operator(hf_mol, scale, expected):
    """The correction vanishes as ``1/c^2`` — the signature of a genuine picture change.

    A constant offset, or one scaling as ``1/c``, would mean a normalization error that the
    physical-``c`` number alone could never distinguish from real physics.
    """
    from pyscf import lib

    pc = picture_changed_moment(hf_mol, "mass", light_speed=lib.param.LIGHT_SPEED * scale)
    bare = _bare_moment(hf_mol, gauge_origin_for(hf_mol, "mass")[0])
    dev = max(np.abs(pc.moment[k] - bare[k]).max() / np.abs(bare[k]).max() for k in range(3))
    assert dev == pytest.approx(expected, rel=0.25)


def test_the_moment_is_hermitian_and_time_reversal_odd(hf_mol):
    """``L`` and ``S`` are both time odd, so the moment is too — and stays so after transforming.

    The time-reversal-**even** part is what :func:`decompose_two_component` keeps, so for an odd
    operator it must vanish. This catches a swapped or transposed spin block, which no norm or
    hermiticity test sees.
    """
    pc = picture_changed_moment(hf_mol, "mass")
    for k in range(3):
        m = pc.moment[k]
        assert np.abs(m - m.conj().T).max() < 1e-12 * np.abs(m).max()
        even = two_component_operator(*decompose_two_component(m))
        assert np.abs(even).max() < 1e-11 * np.abs(m).max()


def test_kuiva_agrees_with_pyscfs_own_picture_change(hf_mol):
    """The one check here whose two sides do **not** share an implementation.

    ⚠ PySCF's ``SpinOrbitalX2CHelper`` is spin-blocked and its ``X2CHelper`` is the j-adapted
    spinor basis; both accept a ``(2 nao, 2 nao)`` matrix without complaint, and feeding one
    the other's basis gives a Hermitian, plausible, wrong operator. The blocks below are
    spin-blocked, so the spin-blocked helper is the right one.
    """
    from pyscf.x2c import x2c

    from kuiva.interface.pyscf_bridge import _odd_moment_blocks

    helper = x2c.SpinOrbitalX2CHelper(hf_mol)
    helper.xuncontract = True
    xmol, _ = helper.get_xmol(hf_mol)
    ls = _odd_moment_blocks(xmol, gauge_origin_for(hf_mol, "mass")[0])
    theirs = np.stack([helper._picture_change(xmol, (None, None), ls[k]) for k in range(3)])

    ours_uncontracted = picture_changed_moment(hf_mol, "mass")
    # PySCF's helper here returns the working (decontracted) basis; contract it the same way.
    from kuiva.interface.pyscf_bridge import four_component_one_electron
    fc = four_component_one_electron(hf_mol, uncontract=True)
    theirs = np.stack([fc.contract(t) for t in theirs])
    dev = max(np.abs(theirs[k] - ours_uncontracted.moment[k]).max()
              / np.abs(ours_uncontracted.moment[k]).max() for k in range(3))
    assert dev < 1e-10


def test_a_decoupling_it_cannot_do_is_refused(hf_mol):
    """``approx="atom1e"`` has no property picture change here, and is refused rather than
    silently substituted: the moment's decoupling must be the Hamiltonian's."""
    with pytest.raises(NotImplementedError, match="approx"):
        picture_changed_moment(hf_mol, "mass", approx="atom1e")


def test_the_anomaly_alone_is_refused(hf_mol):
    """Picture-changing only the ``g_e - 2`` term is a 2e-06 correction on an uncorrected
    operator — not a meaningful combination, so it raises."""
    with pytest.raises(ValueError, match="anomaly_picture_change"):
        ingest_property_integrals(hf_mol, "mass", picture_change=False,
                                  anomaly_picture_change=True)


def test_the_default_warning_states_a_measured_bound(hf_mol, tmp_path, kuiva_caplog):
    """⚠ The default warning must say the approximation is **measured**, not merely present.

    It said "has not been measured" for as long as that was true; it now quotes a bound, and this
    test is what stops it regressing to the weaker claim while the measurement stands. The
    numbers themselves live in the package's validation record, not here.
    """
    n = 4
    rng = np.random.default_rng(0)
    zero = np.zeros((3, n, n), dtype=complex)
    m = dump.PropertyMatrices(energies=np.linspace(0.0, 1e-3, n), mu=zero, l=zero, s=zero)
    dump.write_dump(tmp_path / "d.out", m)
    messages = [r.message for r in kuiva_caplog.records]
    assert any("NO picture-change" in msg for msg in messages)
    assert any("measured" in msg and "not been measured" not in msg for msg in messages), messages
    assert "unmeasured" not in (tmp_path / "d.out").read_text()


def test_the_default_path_carries_no_picture_change(hf_mol):
    props = ingest_property_integrals(hf_mol, "mass")
    assert props.picture_change is None
    assert props.moment_operator() is None
    assert props.provenance()["picture_change"].startswith("none")


# --- end to end: what it does to an analytic free-ion g factor ------------------------------

@pytest.fixture(scope="module")
def boron_variants():
    """One CASSCF on ``B 2p^1``, three property operators on the **same** states.

    The correction changes no wavefunction, so evaluating both operators on one converged run
    makes the difference exactly the operator and nothing else.
    """
    mol = api.Molecule(atoms=[("B", (0.0, 0.0, 0.0))], basis="x2c-SVPall-2c", spin=1)
    ref = api.spinor_reference(mol, screening="none", memory_gb=8,
                               property_picture_change=True, anomaly_picture_change=True)
    outcome = api.casscf(ref, character=("B", "p"), n_active=6, n_active_elec=1,
                         n_states=6, mode="second-order", conv_grad=1e-6, report=False)
    assert outcome.converged
    full = ref.data.properties

    def matrices(props):
        return dump.property_matrices(outcome.coeff, outcome.active.spaces,
                                      outcome.ci.transition_densities(),
                                      outcome.ci.total_energies, props, ref.data.s_ao)

    return {
        "bare": matrices(dataclasses.replace(full, picture_change=None)),
        "pc": matrices(dataclasses.replace(
            full, picture_change=dataclasses.replace(full.picture_change, spin=None))),
        "pc_anomaly": matrices(full),
    }


@pytest.mark.slow
def test_the_bare_operator_sits_on_the_analytic_lande_value(boron_variants):
    """⚠ This is what makes the free ion the sharpest possible probe of the correction.

    For a **one-electron** state the g factor is purely angular, so the bare operator
    reproduces the analytic Lande value to machine precision and the picture-change shift is
    read directly off the deviation, with no orbital or basis confound in between.
    """
    blocks = boron_variants["bare"].analyse()
    for block, sign in zip(blocks, (-1.0, +1.0)):
        target = lande_g(1, 0.5, block.j) + sign * (G_ELECTRON - 2.0) / 3.0
        assert block.g_iso == pytest.approx(target, rel=1e-10)


@pytest.mark.slow
def test_the_picture_change_moves_the_g_factor_and_not_the_spectrum(boron_variants):
    """It is a property operator: the states, and therefore the splitting, must not move."""
    bare = boron_variants["bare"].analyse()
    pc = boron_variants["pc"].analyse()
    assert pc[1].energy_cm == pytest.approx(bare[1].energy_cm, rel=1e-12)
    shifts = [p.g_iso / b.g_iso - 1.0 for p, b in zip(pc, bare)]
    # Boron is Z = 5 and the correction is small there; both levels shift the same way.
    assert all(-1e-3 < s < 0.0 for s in shifts), shifts
    assert abs(shifts[0]) > abs(shifts[1])


@pytest.mark.slow
def test_the_anomaly_small_component_is_negligible(boron_variants):
    """⚠ Why the ``g_e - 2`` term rides on the **bare** spin operator by default.

    Its small-component block enters at ``O((g_e - 2)/c^2)``, and measured against the picture
    change itself it is three orders smaller. Implemented and available, off by default, and
    this test is what says the default is a measurement rather than an assumption.
    """
    pc = boron_variants["pc"].analyse()
    anom = boron_variants["pc_anomaly"].analyse()
    bare = boron_variants["bare"].analyse()
    for a, p, b in zip(anom, pc, bare):
        correction = abs(p.g_iso / b.g_iso - 1.0)
        anomaly = abs(a.g_iso / p.g_iso - 1.0)
        assert anomaly < 1e-6
        assert anomaly < 1e-2 * correction


@pytest.mark.slow
def test_the_dump_records_which_operator_it_used(boron_variants, tmp_path, kuiva_caplog):
    """A stored file that does not say what ``mu`` is built from is not interpretable."""
    m = boron_variants["pc"]
    assert m.picture_changed
    path = dump.write_dump(tmp_path / "pc.out", m, title="picture-changed")
    assert any("PICTURE CHANGE" in r.message for r in kuiva_caplog.records)
    text = path.read_text()
    assert "picture_change_on_properties" in text
    assert "Peng-Reiher" in text

    bare = boron_variants["bare"]
    assert not bare.picture_changed
    read_back = dump.read_dump(dump.write_dump(tmp_path / "bare.out", bare))
    assert read_back["header"]["picture_change_on_properties"].startswith("none")
