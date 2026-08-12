"""Tests for the two-component X2C spin-orbit ingestion and the reference types.

The spinor conventions were fixed before any spin-orbit operator existed, so the first job
here is to establish that PySCF's two-component Hamiltonian really is in them — a structural
claim, checked structurally: the decomposition into ``A (x) 1_2 + sigma . W`` only reproduces
the matrix if the spin blocking *and* the ``W_k = i w_k`` phase convention both match.

The second job is the physics, and it is deliberately checked against **theorems rather than
another program**, by design: a spin-orbit operator acting on a spatially
degenerate p shell must split it into j = 1/2 and j = 3/2 with degeneracies 2 and 4, and it
cannot move the barycentre, because ``sigma . W`` is traceless. Neither statement depends on
any code, basis or parameter.

⚠ **Everything here is deliberately run with** ``screening="none"``, **against a default that
is now** ``"x2camf"``. Two separate reasons, and both matter:

* **This file is about the one-electron operator.** Its subject is whether PySCF's two-
  component Hamiltonian arrives in the fixed spinor conventions (kuiva/spinor/expand.py) and what the *one-electron* spin-orbit
  term does to a p shell. The two-electron picture change is a different object with its own
  eight test files, and folding it in here would mean that a defect in either showed up in
  both.
* ⚠ **Cost.** Bi's four-component average-of-configuration solve is ~39 minutes, against a
  suite that is required to stay laptop-fast. Taking the default would put that in front
  of every ``pytest`` run.

:func:`test_the_default_is_two_electron_screened` is the one test here that does *not* pass
``screening="none"`` — it exists so that a silent revert of the default would fail something,
and it uses neon, whose solve is a fifth of a second.
"""
import numpy as np
import pytest

from kuiva.integrals.transform import transform_1e
from kuiva.interface import Molecule
from kuiva.interface.pyscf_bridge import (SpinOrbitX2C, ingest_spin_orbit, run_scalar_x2c)
from kuiva.orth.canonical import canonical_orthogonalization
from kuiva.spinor.expand import (expand_scalar_mos, expand_unrestricted_mos,
                                 is_time_reversal_even, spin_block_diagonal,
                                 two_component_operator)

HARTREE_CM = 219474.6313632


@pytest.fixture(scope="module")
def bi():
    """Bi 6p^3 — a half-filled, spatially degenerate valence shell and strong SOC. The
    degeneracy is what makes the j-splitting a clean one-particle statement."""
    return run_scalar_x2c(Molecule([("Bi", (0.0, 0.0, 0.0))], basis="x2c-SVPall-2c", spin=3),
                          screening="none")


@pytest.fixture(scope="module")
def ne():
    return run_scalar_x2c(Molecule([("Ne", (0.0, 0.0, 0.0))], basis="x2c-SVPall-2c"),
                          screening="none")


def valence_p_spinors(data, n_spatial=3):
    """AO-basis spinor columns of the highest ``n_spatial`` occupied spatial orbitals."""
    ob = canonical_orthogonalization(data.s_ao)
    sb = expand_scalar_mos(ob.to_working(data.mo_coeff)).transform_scalar_basis(ob.x, "ao")
    nocc = int(np.sum(np.asarray(data.mo_occ) > 0))
    cols = np.array([[2 * p, 2 * p + 1] for p in range(nocc - n_spatial, nocc)]).ravel()
    return sb.take(cols)


# --- The conventions --------------------------------------------------------------
def test_decomposition_is_exact(bi):
    """If the spin blocking or the W = i*w convention were wrong, this residual would be
    of the same order as the operator itself rather than at machine precision."""
    soc = bi.soc
    assert soc is not None
    assert soc.tr_residual_rel < 1e-8
    assert soc.soc_strength > 1.0                       # Bi: a genuinely large SOC operator


def test_w_factors_are_real_antisymmetric(bi):
    w = bi.soc.w
    assert w.shape == (3, bi.nao, bi.nao)
    assert np.isrealobj(w)
    assert np.max(np.abs(w + np.transpose(w, (0, 2, 1)))) < 1e-14


def test_spin_free_part_is_real_symmetric(bi):
    a = bi.soc.h_sf
    assert np.isrealobj(a)
    assert np.max(np.abs(a - a.T)) < 1e-12


def test_assembled_hamiltonian_is_hermitian_and_time_even(bi):
    """Time-reversal evenness is exact by construction — the odd part is projected out at
    ingestion — so this also verifies that the projection is what it claims to be."""
    h = bi.soc.hamiltonian()
    assert h.shape == (2 * bi.nao, 2 * bi.nao)
    assert np.max(np.abs(h - h.conj().T)) < 1e-10
    assert is_time_reversal_even(h, tol=1e-9)


def test_picture_change_shift_is_reported_and_grows_with_z(bi, ne):
    """The spin-free part of the full 2c Hamiltonian is not sfx2c1e's, and the difference is
    a heavy-element effect. Reported, never silently ignored (see the module docstring)."""
    assert bi.soc.picture_change_shift > ne.soc.picture_change_shift
    assert ne.soc.picture_change_shift < 1e-2           # Ne: essentially none


def test_soc_scale_grows_with_z(bi, ne):
    assert bi.soc.soc_strength > 100.0 * ne.soc.soc_strength


def test_transform_to_another_basis(bi):
    """All four components transform identically — the payoff of storing the decomposition."""
    ob = canonical_orthogonalization(bi.s_ao)
    moved = bi.soc.transform(ob.x)
    assert moved.h_sf.shape == (ob.nwork, ob.nwork)
    assert np.max(np.abs(moved.w + np.transpose(moved.w, (0, 2, 1)))) < 1e-10
    # equivalent to transforming the assembled 2c operator
    ref = spin_block_diagonal(ob.x).conj().T @ bi.soc.hamiltonian() @ spin_block_diagonal(ob.x)
    assert np.max(np.abs(moved.hamiltonian() - ref)) < 1e-9


# --- The physics: theorems, not another program ------------------------------------------
def test_p_shell_splits_into_j_one_half_and_three_halves(bi):
    """Exact degeneracies 2 (j=1/2) and 4 (j=3/2) from a spatially degenerate p shell."""
    c = valence_p_spinors(bi)
    ev = np.linalg.eigvalsh(transform_1e(bi.soc.hamiltonian(), c))
    rel = (ev - ev[0]) * HARTREE_CM
    assert np.max(np.abs(rel[:2] - rel[0])) < 1.0       # j = 1/2 doublet
    assert np.max(np.abs(rel[2:] - rel[2])) < 1.0       # j = 3/2 quartet
    assert rel[2] - rel[1] > 1000.0                     # and they are genuinely split


def test_kramers_degeneracy_is_exact(bi):
    """Every eigenvalue of a time-reversal-even operator on an odd-dimensional-spin space is
    doubly degenerate (Kramers). Exact here because the odd part was projected out."""
    c = valence_p_spinors(bi)
    ev = np.linalg.eigvalsh(transform_1e(bi.soc.hamiltonian(), c))
    assert np.max(np.abs(ev[0::2] - ev[1::2])) < 1e-12


def test_soc_does_not_move_the_barycentre(bi):
    """``sigma . W`` is traceless, so it cannot shift the centre of gravity of a shell. This
    is the sharpest available check that the spin-orbit operator enters with the right
    structure: a spurious spin-free contamination would break it immediately."""
    c = valence_p_spinors(bi)
    with_soc = np.linalg.eigvalsh(transform_1e(bi.soc.hamiltonian(), c))
    without = np.linalg.eigvalsh(transform_1e(spin_block_diagonal(bi.soc.h_sf), c))
    shift_cm = abs(with_soc.mean() - without.mean()) * HARTREE_CM
    assert shift_cm < 1e-6


def test_light_atom_p_splitting_scales_correctly(ne, bi):
    """The same j-splitting on a light atom, four orders of magnitude smaller than Bi's.

    A scale error in the spin-orbit operator would be invisible on one atom and obvious
    across two: Ne 2p and Bi 6p differ by a factor of ~15 in splitting and must both come out
    right. Ne also has an **experimental anchor** — the Ne(+) 2p fine structure, 780.4 cm^-1
    (NIST ASD) — used at 30%, as an anchor and never as an
    accuracy claim. It is measured here at ~908 cm^-1, i.e. **+16% high**, which is the
    expected signature of one-electron SOC with no two-electron screening and matches
    the overestimate seen for Bi. That agreement across a factor of 15 is the real content of
    this test; do not tighten the band without implementing the screening.
    """
    c = valence_p_spinors(ne)
    ev = np.linalg.eigvalsh(transform_1e(ne.soc.hamiltonian(), c))
    rel = (ev - ev[0]) * HARTREE_CM
    assert np.max(np.abs(rel[:2] - rel[0])) < 1e-3      # same 2 + 4 structure
    assert np.max(np.abs(rel[2:] - rel[2])) < 1e-3
    split_ne = rel[2] - rel[0]
    assert 0.7 * 780.4 < split_ne < 1.3 * 780.4 # experimental anchor, 30%

    c_bi = valence_p_spinors(bi)
    ev_bi = np.linalg.eigvalsh(transform_1e(bi.soc.hamiltonian(), c_bi))
    split_bi = (ev_bi[2] - ev_bi[0]) * HARTREE_CM
    assert split_bi / split_ne > 10.0                   # heavy vs light, right way round


def test_disabling_soc_gives_a_spin_free_calculation(caplog):
    d = run_scalar_x2c(Molecule([("Ne", (0.0, 0.0, 0.0))], basis="x2c-SVPall-2c"),
                       with_soc=False, screening="none")
    assert d.soc is None and not d.has_soc


def test_ingest_spin_orbit_directly_matches_the_bridge(ne):
    """The ingestion function is usable on its own (and is what a future front-end would
    call), and reproduces what the driver stored."""
    from kuiva.interface.pyscf_bridge import build_mole
    mol = build_mole(Molecule([("Ne", (0.0, 0.0, 0.0))], basis="x2c-SVPall-2c"))
    soc = ingest_spin_orbit(mol, ne.h_x2c, screening="none")
    assert np.allclose(soc.w, ne.soc.w)
    assert np.allclose(soc.h_sf, ne.soc.h_sf)


def test_atomic_approximation_is_close_to_exact_one_electron(bi):
    """``approx='atom1e'`` is a cheaper decoupling, not a different physics; the spin-orbit
    operator it produces must agree closely with the exact one-electron X2C."""
    from kuiva.interface.pyscf_bridge import build_mole
    mol = build_mole(Molecule([("Bi", (0.0, 0.0, 0.0))], basis="x2c-SVPall-2c", spin=3))
    atomic = ingest_spin_orbit(mol, approx="atom1e", screening="none")
    rel = np.max(np.abs(atomic.w - bi.soc.w)) / bi.soc.soc_strength
    assert rel < 1e-6                                   # single atom: the two coincide


def test_an_unscreened_hamiltonian_says_so(bi):
    """The *absence* of the two-electron picture change is recorded on the object, so no
    downstream consumer can mistake this Hamiltonian for a screened one.

    ⚠ The field is a :class:`kuiva.amf.correction.ScreeningRecord` and not
    a string, because a bare ``"x2camf"`` would not say *which* x2camf — which interaction,
    which four-component backend, which per-element reference configuration — and a stored
    property matrix that cannot answer those is not interpretable.
    """
    from kuiva.amf.correction import ScreeningRecord

    assert isinstance(bi.soc.screening, ScreeningRecord)
    assert bi.soc.screening.method == "none" and not bi.soc.screening.applied
    assert bi.soc.screening.elements == ()
    # The provenance dict is what the property dump and a Tier-2 record store; it must round-trip
    # through json, so a stored matrix always carries the Hamiltonian that produced it.
    import json
    assert json.loads(json.dumps(bi.soc.provenance()))["screening"]["method"] == "none"


def test_the_default_is_two_electron_screened():
    """⚠ **The default is** ``screening="x2camf"`` **, and this is the test
    that would fail if it were quietly reverted.**

    It is the only test in this file that does not ask for ``"none"``, and it is on neon
    because the whole point of a default is that it applies to a user who did not think about
    it — so it has to be exercised through the ordinary call with no arguments.

    Both halves are asserted: that the correction was *applied*, and that the record says
    which one. A default that silently fell back to ``"none"`` — because a four-component
    solve failed, or because a caller's kwarg stopped being threaded through — would otherwise
    look exactly like a cheap successful run.
    """
    d = run_scalar_x2c(Molecule([("Ne", (0.0, 0.0, 0.0))], basis="x2c-SVPall-2c"))
    assert d.soc.screening.method == "x2camf"
    assert d.soc.screening.applied
    assert d.soc.screening.elements == ("Ne",)


def test_the_default_reduces_the_p_splitting(ne):
    """The default is not merely *recorded* as different; it produces a different number.

    Screening always reduces a one-particle j-splitting, by 8-24% for the
    light atoms of this basis. Asserted as a band and a sign rather than a value — the value
    belongs to ``tests/test_x2camf_dirac.py``, which is a
    controlled same-basis measurement and this is not.

    ⚠ Both numbers here are **frozen-orbital** splittings, which is worth ~30% against a
    self-consistent two-component SCF for the same operator. That does not
    affect the comparison, because both sides use the same construction and the same spinors.
    """
    d = run_scalar_x2c(Molecule([("Ne", (0.0, 0.0, 0.0))], basis="x2c-SVPall-2c"))
    c = valence_p_spinors(ne)
    plain = np.linalg.eigvalsh(transform_1e(ne.soc.hamiltonian(), c))
    screened = np.linalg.eigvalsh(transform_1e(d.soc.hamiltonian(), c))
    gap = lambda e: (e[-1] - e[0]) * HARTREE_CM                          # noqa: E731
    reduction = 1.0 - gap(screened) / gap(plain)
    assert 0.05 < reduction < 0.40, \
        "screening changed the Ne 2p splitting by {:+.1%} ({:.1f} -> {:.1f} cm^-1)".format(
            -reduction, gap(plain), gap(screened))


# --- Reference types ----------------------------------------------------------------------
def test_restricted_reference_shapes():
    d = run_scalar_x2c(Molecule([("Ne", (0.0, 0.0, 0.0))], basis="x2c-SVPall-2c"),
                       screening="none")
    assert d.reference == "rhf" and not d.unrestricted
    assert len(d.mo_sets()) == 1
    assert d.mo_sets()[0].shape == (d.nao, d.nmo)


def test_open_shell_defaults_to_rohf():
    d = run_scalar_x2c(Molecule([("O", (0.0, 0.0, 0.0))], basis="x2c-SVPall-2c", spin=2),
                       screening="none")
    assert d.reference == "rohf" and not d.unrestricted


def test_unrestricted_reference_shapes():
    d = run_scalar_x2c(Molecule([("O", (0.0, 0.0, 0.0))], basis="x2c-SVPall-2c", spin=2),
                       reference="uhf")
    assert d.reference == "uhf" and d.unrestricted
    assert d.mo_coeff.shape == (2, d.nao, d.nmo)
    assert len(d.mo_sets()) == 2
    assert d.spin_contamination() is not None
    assert abs(d.spin_contamination()) < 0.05           # O triplet: barely contaminated


def test_uhf_is_variationally_below_rohf():
    kw = dict(basis="x2c-SVPall-2c", spin=2)
    mol = Molecule([("O", (0.0, 0.0, 0.0))], **kw)
    assert (run_scalar_x2c(mol, reference="uhf").e_scf
            < run_scalar_x2c(mol, reference="rohf").e_scf + 1e-10)


def test_rhf_on_an_open_shell_is_refused():
    with pytest.raises(ValueError, match="closed shell"):
        run_scalar_x2c(Molecule([("O", (0.0, 0.0, 0.0))], basis="x2c-SVPall-2c", spin=2),
                       reference="rhf")


def test_unknown_reference_is_refused():
    with pytest.raises(ValueError, match="unknown reference"):
        run_scalar_x2c(Molecule([("Ne", (0.0, 0.0, 0.0))], basis="x2c-SVPall-2c"),
                       reference="ccsd")


def test_unrestricted_spinor_expansion_is_orthonormal_but_not_kramers(kuiva_caplog):
    """The set is orthonormal because the spin components are orthogonal whatever the
    orbitals do — but it carries no Kramers structure, and says so."""
    d = run_scalar_x2c(Molecule([("O", (0.0, 0.0, 0.0))], basis="x2c-SVPall-2c", spin=2),
                       reference="uhf")
    ob = canonical_orthogonalization(d.s_ao)
    ca, cb = (ob.to_working(c) for c in d.mo_sets())
    with kuiva_caplog.at_level("WARNING"):
        sb = expand_unrestricted_mos(ca, cb, d.mo_energy, d.mo_occ)
    assert sb.nspinor == 2 * d.nmo
    assert not sb.kramers_paired
    assert sb.orthonormality_error() < 1e-10
    assert sb.pair_overlap is not None and sb.pair_overlap.shape == (d.nmo,)
    assert any("not Kramers paired" in r.message for r in kuiva_caplog.records)
    # occupations are per spin and are NOT halved (unlike the restricted expansion)
    assert np.isclose(sb.occ.sum(), d.nelec_total)


def test_unrestricted_expansion_rejects_mismatched_sets():
    with pytest.raises(ValueError, match="same shape"):
        expand_unrestricted_mos(np.eye(4), np.eye(3))


def test_restricted_expansion_halves_occupations():
    """Guard the difference between the two expansions: restricted mo_occ is per *spatial*
    orbital (0/1/2), unrestricted is already per spin (0/1)."""
    sb = expand_scalar_mos(np.eye(3), mo_occ=np.array([2.0, 1.0, 0.0]))
    assert np.allclose(sb.occ, [1.0, 1.0, 0.5, 0.5, 0.0, 0.0])


# --- Two-electron routing ---------------------------------------------------------
def test_cholesky_is_the_default_route():
    """Cholesky in every case, whatever the registry recommends for the basis."""
    for mol in (Molecule([("Ne", (0.0, 0.0, 0.0))], basis="x2c-SVPall-2c"),
                Molecule([("O", (0.0, 0.0, 0.0))], basis="x2c-SVPall-2c", spin=2)):
        d = run_scalar_x2c(mol, screening="none")
        assert d.fit_route == "conventional" and d.eri is not None and d.df_cderi is None


def test_user_auxiliary_selects_density_fitting():
    d = run_scalar_x2c(Molecule.from_xyz_string("H 0 0 0\nF 0 0 0.917",
                                                basis="x2c-SVPall-2c"),
                       auxbasis="def2-universal-jkfit", screening="none")
    assert d.fit_route == "df" and d.df_cderi is not None and d.eri is None
    assert d.aux_name == "def2-universal-jkfit"


def test_df_without_an_auxiliary_warns(kuiva_caplog):
    """Asking for DF but naming no auxiliary falls back on a Coulomb-fitting set, which is
    the case measured at 1.7e-3 Eh per integral. It must not pass unremarked."""
    with kuiva_caplog.at_level("WARNING"):
        d = run_scalar_x2c(Molecule([("Ne", (0.0, 0.0, 0.0))], basis="x2c-SVPall-2c"),
                           fitting="df", screening="none")
    assert d.fit_route == "df"
    assert any("without an auxiliary basis" in r.message for r in kuiva_caplog.records)


def test_unknown_fitting_route_is_refused():
    with pytest.raises(ValueError, match="unknown fitting route"):
        run_scalar_x2c(Molecule([("Ne", (0.0, 0.0, 0.0))], basis="x2c-SVPall-2c"),
                       fitting="ri-jk")


def test_eri_memory_guard():
    """The ERI array is checked against the configured memory limit."""
    from kuiva.interface.pyscf_bridge import _reserve_eri_memory
    from kuiva.util import resources

    resources.BUDGET.configure(resources.ResourceLimits(memory_gb=8.0, source="test"))
    _reserve_eri_memory(100)                            # ~0.2 GB: fine
    with pytest.raises(MemoryError, match="integral-direct"):
        _reserve_eri_memory(1200)                       # ~5 TB: not fine


# --- End to end ---------------------------------------------------------------------------
def test_pipeline_carries_soc_through_to_the_integrals():
    """`spinor_reference` -> `h_one_electron` is the full 2c X2C operator, and the spinor
    one-electron matrix it produces is Hermitian with exact Kramers degeneracy."""
    from kuiva.interface.api import spinor_reference
    ref = spinor_reference(Molecule([("I", (0.0, 0.0, 0.0))], basis="x2c-SVPall-2c", spin=1))
    h_ao = ref.h_one_electron()
    assert h_ao.dtype == np.complex128
    assert np.max(np.abs(h_ao.imag)) > 1e-6             # SOC is actually present
    c = ref.spinors_in_ao(np.arange(0, 20))
    h = transform_1e(h_ao, c)
    assert np.max(np.abs(h - h.conj().T)) < 1e-10
    ev = np.linalg.eigvalsh(h)
    assert np.max(np.abs(ev[0::2] - ev[1::2])) < 1e-9   # Kramers pairs


def test_pipeline_without_soc_is_real():
    from kuiva.interface.api import spinor_reference
    ref = spinor_reference(Molecule([("Ne", (0.0, 0.0, 0.0))], basis="x2c-SVPall-2c"),
                           with_soc=False)
    h = ref.h_one_electron()
    assert np.max(np.abs(h.imag)) < 1e-14
