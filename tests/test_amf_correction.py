"""Tests for the top-level AMF seam and its integration.

Three things:

* **The seam.** ``method="none"`` must leave the Hamiltonian bitwise what it was before any of
  this existed. That is what makes the correction safe to land before it is the default.
* **The structural validators**, which every later stage relies on, tested by *breaking* each
  invariant in turn — a validator that has never been seen to fail is not known to work.
* **The physics, end to end**: the atomic j-splitting against the four-component reference and
  against experiment, which is the number this whole plan exists to fix.
"""
import numpy as np
import pytest

from kuiva.amf import amf_correction, validate_correction
from kuiva.amf.atomic import (atomic_solution, basis_digest, cache_key, cache_statistics,
                              clear_cache, make_request)
from kuiva.amf.correction import AMFCorrection, zero_correction
from kuiva.integrals.transform import transform_1e
from kuiva.interface.pyscf_bridge import build_mole, ingest_spin_orbit
from kuiva.orth.canonical import canonical_orthogonalization
from kuiva.spinor.expand import (expand_scalar_mos, is_time_reversal_even,
                                 two_component_operator)

BASIS = "x2c-SVPall-2c"
HARTREE_CM = 219474.6313632
#: Ne(+) 2p fine structure, NIST ASD. Used as an anchor,
#: never an accuracy claim.
NE_EXPERIMENT_CM = 780.4


def atom(symbol, basis=BASIS, charge=0, spin=0):
    from pyscf import gto
    return gto.M(atom=[(symbol, (0.0, 0.0, 0.0))], basis=basis, charge=charge, spin=spin,
                 verbose=0)


def frozen_orbital_splitting(mol, correction=None):
    """Valence-p j-splitting [cm^-1] from the **frozen scalar orbitals** of an sfx2c1e SCF.

    The construction of ``tests/test_soc_ingestion.py``, reproduced deliberately so that the
    uncorrected number here is the *same* number that file records (908 cm^-1 for Ne). It is a
    one-particle expectation value in a fixed six-dimensional space: no orbital relaxation,
    and in particular no response to the large **spin-free** part of the correction.
    """
    from pyscf import scf

    # ⚠ ``screening="none"`` explicitly, against a default that is now "x2camf"
    #. This helper's contract is *the bare one-electron operator, plus
    # whatever the caller passes* — taking the default here would add the correction twice
    # and halve the splitting, which is exactly the double-counting the record exists to prevent.
    h = ingest_spin_orbit(mol, screening="none").hamiltonian()
    if correction is not None:
        h = h + correction.hamiltonian()
    mf = scf.RHF(mol).sfx2c1e()
    mf.verbose = 0
    mf.kernel()
    ob = canonical_orthogonalization(mol.intor("int1e_ovlp"))
    sb = expand_scalar_mos(ob.to_working(mf.mo_coeff)).transform_scalar_basis(ob.x, "ao")
    nocc = int(np.sum(np.asarray(mf.mo_occ) > 0))
    cols = np.array([[2 * p, 2 * p + 1] for p in range(nocc - 3, nocc)]).ravel()
    ev = np.linalg.eigvalsh(transform_1e(h, sb.take(cols)))
    return float(ev[2] - ev[0]) * HARTREE_CM


def self_consistent_spectrum(mol, correction=None):
    """Occupied spinor energies [Eh] from a **self-consistent two-component** SCF, ascending.

    ⚠ **This, and not the frozen-orbital construction above, is what compares with a
    four-component reference**, because the four-component reference is itself self-consistent.
    The distinction is not pedantic and it cost a debugging session: the two constructions give
    1174 and 908 cm^-1 for the *same uncorrected operator* on neon. The frozen-orbital value is
    lower because it evaluates the two-component Hamiltonian in orbitals optimized without it,
    and it happens to land near the four-component answer for reasons that have nothing to do
    with two-electron screening.

    Both constructions agree on what the correction *does* — each reduces the splitting by
    about 23% — and that agreement is asserted below. What only the self-consistent one gets
    right is the absolute value. Kuiva's own use is a CASSCF, where the orbitals are optimized
    in the presence of the full Hamiltonian, so the self-consistent construction is the
    relevant one.

    ``tests/test_x2camf_dirac.py`` imports this rather than writing a second copy: the
    construction is part of what a quoted splitting *means*, so there must be
    exactly one of it in the suite.
    """
    import scipy.linalg
    from pyscf import scf

    # ⚠ ``screening="none"``, for the reason given in :func:`frozen_orbital_splitting`: the
    # correction is the caller's to pass, and taking the now-screened default would add it
    # twice. Both helpers were caught by this when the default flipped, and both are the
    # shape of code most likely to be copied into a new comparison.
    h = ingest_spin_orbit(mol, screening="none").hamiltonian()
    if correction is not None:
        h = h + correction.hamiltonian()
    s = mol.intor_symmetric("int1e_ovlp")
    mf = scf.GHF(mol)
    mf.verbose = 0
    mf.conv_tol = 1e-11
    mf.max_cycle = 300
    mf.get_hcore = lambda *a, **k: h
    mf.get_ovlp = lambda *a, **k: scipy.linalg.block_diag(s, s)
    mf.kernel()
    assert mf.converged
    return np.sort(np.asarray(mf.mo_energy))[:mol.nelectron]


def self_consistent_splitting(mol, correction=None):
    """Valence-p j-splitting [cm^-1] from :func:`self_consistent_spectrum` — the spread of the
    six frontier occupied spinors, i.e. ``p_1/2`` (2) against ``p_3/2`` (4)."""
    occupied = self_consistent_spectrum(mol, correction)[-6:]
    return float(occupied[-1] - occupied[0]) * HARTREE_CM


@pytest.fixture(scope="module")
def ne_correction():
    clear_cache()
    return atom("Ne"), amf_correction(atom("Ne"), method="x2camf")


# --- The seam ---------------------------------------------------------------

def test_method_none_is_exactly_zero():
    """Exact zeros, not small numbers — so that adding the correction is a no-op at the level
    of floating-point bits and not merely of tolerances."""
    mol = atom("Ne")
    c = amf_correction(mol, method="none")
    assert c.is_zero
    assert c.h_sf.shape == (mol.nao, mol.nao)
    assert c.w.shape == (3, mol.nao, mol.nao)
    assert not c.h_sf.any() and not c.w.any()
    assert c.method == "none" and c.interaction == "none"


def test_method_none_leaves_the_hamiltonian_bitwise_identical():
    """The property that lets this land before it is the default. ``==`` and not ``allclose``:
    anything else would let a 1-ulp change through,
    and 1 ulp on a 1e4 Eh heavy-element matrix element is not nothing."""
    mol = atom("Ne")
    soc = ingest_spin_orbit(mol, screening="none")
    correction = amf_correction(mol, method="none")
    corrected = soc.hamiltonian() + correction.hamiltonian()
    assert np.array_equal(corrected, soc.hamiltonian())


def test_x2camf_is_the_default_method():
    """X2CAMF is the default here as it is at the front-end seam.

    ⚠ This test used to assert the opposite, and the inversion is a user decision.
    It is kept pointing at the same function rather than deleted, because the property worth
    guarding is that ``amf_correction`` and ``ingest_spin_orbit`` **agree** on what "no
    argument" means. Two seams with two different defaults is how a caller ends up correcting
    a Hamiltonian twice or not at all.
    """
    correction = amf_correction(atom("Ne"))
    assert not correction.is_zero
    assert correction.method == "x2camf"
    assert amf_correction(atom("Ne"), method="none").is_zero


def test_unknown_method_and_interaction_are_refused():
    mol = atom("Ne")
    with pytest.raises(ValueError, match="unknown two-electron picture-change method"):
        amf_correction(mol, method="snso")
    with pytest.raises(ValueError, match="unknown two-electron interaction"):
        amf_correction(mol, method="x2camf", interaction="gaunt-breit")


def test_ecp_molecule_is_refused():
    """X2C has no meaning with a pseudopotential core, and this is the door the check has to
    guard: a backend receives only an element's parsed basis *functions*, which carry no ECP,
    so by the time the atomic solve starts the pseudopotential is invisible."""
    from pyscf import gto
    mol = gto.M(atom="I 0 0 0", basis="def2-svp", ecp="def2-svp", spin=1, verbose=0)
    assert mol.has_ecp()
    with pytest.raises(NotImplementedError, match="all-electron"):
        amf_correction(mol, method="x2camf")


def test_one_electron_system_gives_exactly_zero(kuiva_caplog):
    """A one-electron atom has no second electron to screen, so there is no two-electron mean
    field and no picture change of one. Exactly zero **by definition** — see the reasoning in
    :func:`kuiva.amf.correction.amf_correction`; what a Hartree-Fock mean field would give
    here is pure self-interaction, an artefact of the method rather than physical screening.

    Because it is a definition, it cannot on its own test the subtraction. The test that does
    is ``test_zero_mean_field_gives_exactly_zero_correction`` in ``test_amf_decouple.py``,
    which drives the machinery itself.
    """
    c = amf_correction(atom("He", charge=1, spin=1), method="x2camf")
    assert c.is_zero and c.method == "x2camf"


# --- The structural validators -------------------------------

def test_validator_accepts_a_real_correction(ne_correction):
    _, c = ne_correction
    validate_correction(c.h_sf, c.w)


def test_validator_rejects_each_broken_invariant():
    """Every invariant broken in turn. A validator that has never been observed to fail is
    not known to work, and this one guards the property that keeps Kramers degeneracy
    exact at 1e-8 — a failure of which would not surface until the CI."""
    n = 4
    h = np.zeros((n, n))
    w = np.zeros((3, n, n))
    validate_correction(h, w)                                     # the trivial case is valid

    bad = h.copy()
    bad[0, 1] = 1.0                                               # not symmetric
    with pytest.raises(ValueError, match="not symmetric"):
        validate_correction(bad, w)

    bad_w = w.copy()
    bad_w[2, 0, 1] = bad_w[2, 1, 0] = 1.0                         # symmetric, not anti
    with pytest.raises(ValueError, match="not antisymmetric"):
        validate_correction(h, bad_w)

    with pytest.raises(ValueError, match="not real"):
        validate_correction(h.astype(complex) + 1j * np.eye(n), w)

    with pytest.raises(ValueError, match="shape"):
        validate_correction(h, np.zeros((2, n, n)))


def test_validated_correction_is_time_reversal_even(ne_correction):
    _, c = ne_correction
    h = c.hamiltonian()
    assert np.max(np.abs(h - h.conj().T)) < 1e-12
    assert is_time_reversal_even(h, tol=1e-12)


def test_zero_correction_helper():
    z = zero_correction(5)
    assert isinstance(z, AMFCorrection) and z.is_zero and z.nao == 5
    validate_correction(z.h_sf, z.w)


# --- Provenance ----------------------------------------------------------------------------

def test_provenance_is_recorded(ne_correction):
    """A stored property matrix must never be ambiguous about which Hamiltonian produced it.
    Everything needed to say so is on the object."""
    _, c = ne_correction
    assert c.method == "x2camf"
    assert c.interaction == "coulomb"
    assert c.backend == "pyscf" and c.backend_version
    assert c.elements == ("Ne",)
    assert c.light_speed is None                    # physical
    assert c.spin_free_scale > 0.0 and c.spin_orbit_scale > 0.0


def test_report_runs_at_both_settings(ne_correction, kuiva_caplog):
    """The standard output block, for the corrected and the uncorrected Hamiltonian alike. The
    uncorrected one must say so in the output, not stay silent."""
    import logging

    mol, c = ne_correction
    with kuiva_caplog.at_level(logging.INFO):
        c.report()
        amf_correction(mol, method="none").report()
    text = "\n".join(r.message for r in kuiva_caplog.records)
    assert "x2camf" in text and "none" in text


def test_relative_size_against_the_one_electron_operator(ne_correction):
    """The spin-orbit correction is a few per cent of the one-electron spin-orbit operator —
    which is the magnitude that a 15-30% error in the *splitting* corresponds to. A correction
    of a few per mille or of tens of per cent would both mean something is wrong."""
    mol, c = ne_correction
    sf, so = c.relative_to(ingest_spin_orbit(mol, screening="none"))
    assert 1e-5 < sf < 1e-1
    assert 1e-3 < so < 5e-1


def test_light_speed_override_is_recorded_and_warned_about(kuiva_caplog):
    """A correction built at a non-physical ``c`` is a numerical experiment. It must be
    impossible to mistake one for a calculation."""
    import logging

    clear_cache()
    c = amf_correction(atom("Ne"), method="x2camf", light_speed=1.0e5)
    assert c.light_speed == 1.0e5
    with kuiva_caplog.at_level(logging.WARNING):
        c.report()
    assert any("NON-PHYSICAL" in r.message for r in kuiva_caplog.records)


# --- Caching --------------------------------------------------

@pytest.mark.stage_under_test("amf_atomic")   # asserts a SOLVE COUNT: no cache may serve it
def test_repeated_requests_hit_the_cache():
    """Verified by **call count**, never by timing: on this machine wall time is partly a
    measure of the thermal envelope."""
    clear_cache()
    mol = atom("Ne")
    first = amf_correction(mol, method="x2camf")
    assert cache_statistics()["solves"] == 1
    second = amf_correction(mol, method="x2camf")
    assert cache_statistics()["solves"] == 1
    assert cache_statistics()["correction_hits"] == 1
    assert np.array_equal(first.h_sf, second.h_sf)


def test_the_speed_of_light_is_part_of_the_cache_key():
    """⚠ The trap this key exists to avoid: the non-relativistic-limit test runs the same
    element in the same basis at a different ``c``, and a cache that ignored it would return
    the physical correction and the test would pass for the wrong reason — the worst possible
    failure of a test whose whole job is to be unfoolable."""
    a = make_request("Ne", "x2c-SVPall-2c")
    b = make_request("Ne", "x2c-SVPall-2c", light_speed=1.0e5)
    assert a != b and hash(a) != hash(b)
    assert cache_key(a) != cache_key(b)
    assert "c-physical" in cache_key(a) and "c1.0" in cache_key(b)


def test_the_basis_is_keyed_by_content_not_by_name():
    """Two requests naming the same family but carrying different parsed data are different
    calculations. Keying on the name would silently reuse a correction built for different
    functions."""
    mol_a, mol_b = atom("Ne", BASIS), atom("Ne", "cc-pvdz")
    assert basis_digest(mol_a._basis["Ne"]) != basis_digest(mol_b._basis["Ne"])
    assert basis_digest(mol_a._basis["Ne"]) == basis_digest(mol_a._basis["Ne"])


def test_interaction_and_configuration_separate_cache_entries():
    """Every axis that changes the answer must change the key.

    ⚠ ``charge`` is deliberately **not** among them any more: the reference configuration is
    the single source of truth for the charge state, and ``make_request`` derives the charge
    from it (:mod:`kuiva.amf.configuration`). Keying on the *requested* charge would let two
    requests that produce the identical calculation occupy two entries, and would label the
    stored result with a number that does not describe it.
    """
    base = make_request("Ne", "b")
    assert base != make_request("Ne", "b", interaction="gaunt")
    assert base != make_request("Ne", "b", uncontract=False)
    assert base != make_request("Ne", "b", light_speed=1370.0)
    # A different reference configuration is a different calculation...
    assert base != make_request("Ne", "b", configuration="1s2 2s2 2p4")
    # ...and the charge alone is not, because it no longer decides anything.
    assert base == make_request("Ne", "b", charge=2)


def test_two_spellings_of_one_configuration_share_a_cache_entry():
    """The other half of the same requirement, and the one a free-form string gets wrong.

    ``"[Ar]3d1"`` and the shell-by-shell spelling describe the same reference state, so they
    must hash and compare equal — otherwise the sensitivity study of the open-shell path pays for
    every solve twice and the caching that makes an atomic mean field affordable at all stops
    working for exactly the ions it matters for.
    """
    a = make_request("Ti", "b", configuration="[Ar]3d1")
    b = make_request("Ti", "b", configuration="1s2 2s2 2p6 3s2 3p6 3d1")
    assert a == b and hash(a) == hash(b)
    assert a.charge == 3                      # derived from the configuration, not passed in
    assert cache_key(a) == cache_key(b)


# --- The physics, end to end ----------------------------------------------------------------

def test_neon_j_splitting_moves_from_thirty_percent_high_to_the_reference():
    """**The number this plan exists to fix.**

    The one-electron X2C operator overestimates the atomic j-splitting, and the four-component
    Dirac-Coulomb result for the *same atom in the same basis* is the reference that says by
    how much. Measured here: 1174 cm^-1 one-electron against 903 cm^-1 four-component, i.e.
    **+30%**; with the atomic mean field, 906 cm^-1, i.e. **+0.3%**.

    The band asserted is deliberately wide (a factor of five improvement, and within 10% of
    the reference) because this is a regression guard, not a fit. The tight statement lives in
    ``test_amf_decouple.py``, where the *energy functional* is compared at 3e-07 Eh.
    """
    clear_cache()
    mol = atom("Ne")
    correction = amf_correction(mol, method="x2camf")
    # The reference is computed, not stored: the four-component j-splitting of the same shell
    # of the same atom in the same basis, read off the atomic solution's spinor energies.
    reference = atomic_solution("Ne", mol._basis["Ne"]).shell_splitting() * HARTREE_CM

    before = self_consistent_splitting(mol)
    after = self_consistent_splitting(mol, correction)

    assert 850.0 < reference < 950.0                     # the 4c answer, sanity-bounded
    assert before > 1.2 * reference                      # the known one-electron overestimate
    assert after < before                                # screening reduces it, always
    assert abs(after - reference) < 0.10 * reference     # and lands on the 4c reference
    assert abs(before - reference) > 5.0 * abs(after - reference)


def test_both_constructions_agree_on_what_the_correction_does():
    """⚠ The trap recorded in :func:`self_consistent_splitting`, asserted so it stays recorded.

    The frozen-orbital and self-consistent constructions disagree by 30% on the *absolute*
    splitting of the uncorrected operator (908 against 1174 cm^-1 for neon) because one of them
    evaluates the Hamiltonian in orbitals optimized without it. They nonetheless agree closely
    on the **fractional reduction** the correction produces, ~23%, because that is a property
    of the operator rather than of the orbitals.

    Anyone comparing a Kuiva splitting against a published one must know which construction
    produced it. This test exists so that fact is discoverable from the suite.
    """
    clear_cache()
    mol = atom("Ne")
    correction = amf_correction(mol, method="x2camf")
    frozen = (frozen_orbital_splitting(mol), frozen_orbital_splitting(mol, correction))
    scf_based = (self_consistent_splitting(mol), self_consistent_splitting(mol, correction))

    assert scf_based[0] > 1.2 * frozen[0]                 # the constructions genuinely differ
    reductions = [1.0 - after / before for before, after in (frozen, scf_based)]
    assert all(0.15 < r < 0.35 for r in reductions)       # both ~23%
    assert abs(reductions[0] - reductions[1]) < 0.05      # and they agree on it


def test_neon_j_splitting_against_experiment():
    """The NIST anchor, used as an anchor — never an accuracy
    claim, at 30%.

    ⚠ Do not tighten this band without adding the Gaunt term *and* an ionic reference
    configuration. The residual disagreement here is not the picture change (which
    ``test_amf_decouple.py`` pins at 3e-07 Eh against four-component theory): it is that the
    experimental number is the fine structure of **Ne(+)**, a 2p^5 hole state, while the mean
    field is taken over **neutral** Ne, which has one more electron doing the screening. That
    sensitivity is measured by the open-shell study, not a tolerance to tune here.
    """
    clear_cache()
    mol = atom("Ne")
    after = self_consistent_splitting(mol, amf_correction(mol, method="x2camf"))
    assert 0.7 * NE_EXPERIMENT_CM < after < 1.3 * NE_EXPERIMENT_CM


@pytest.mark.stage_under_test("amf_atomic")   # asserts a SOLVE COUNT: no cache may serve it
def test_gaunt_gives_a_different_correction():
    """A different two-electron interaction in the atomic reference is a different
    calculation, recorded as one."""
    clear_cache()
    mol = atom("Ne")
    coulomb = amf_correction(mol, method="x2camf", interaction="coulomb")
    gaunt = amf_correction(mol, method="x2camf", interaction="gaunt")
    assert gaunt.interaction == "gaunt"
    assert not np.allclose(coulomb.w, gaunt.w)
    assert cache_statistics()["solves"] == 2
