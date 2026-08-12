"""Tier-2 tests: spin-orbit references from OpenMolcas and DIRAC.

Tier 2 compares across *programs*, so it must be built around the fact that Kuiva,
OpenMolcas and DIRAC will never agree to many digits: the Hamiltonians differ (X2C at the CI
step vs DKH2+AMFI vs DIRAC's X2C/4c), the orbitals differ (one state-averaged spinor set vs
per-multiplicity RASSCF orbitals), and the dump fixes no phase convention. Tests that demand
numerical agreement across codes are worse than useless - they fail for correct code and get
silenced.

So the assertions are graded by how well defined the quantity actually is:

======================  =============  ==================================================
quantity                tolerance      why it is meaningful
======================  =============  ==================================================
degeneracy pattern      **exact**      fixed by symmetry; no method can change it
Lande g of a free ion   1%             analytic; independent of every code here
g isotropy (free ion)   1e-3           spherical symmetry; a pure internal consistency check
SOC splittings          15%            genuinely method-dependent (DKH2+AMFI vs X2C)
tensor-product energies 1 cm^-1        exact by locality at 25 A, within one code
absolute energies       not compared   different Hamiltonians; stored only as provenance
======================  =============  ==================================================

The reference files are committed, so none of this needs OpenMolcas or DIRAC installed.

⚠ **The 15% band was revisited when X2CAMF landed and deliberately not
tightened.** It is the obvious move — the method error the correction removes is 5-30%, of the
same size as the band — and it is wrong here for three reasons, recorded
and summarized so nobody has to rediscover them:

1. **This band does not measure Kuiva.** It compares OpenMolcas against DIRAC, which differ in
   the Hamiltonian and in the orbitals; nothing Kuiva does moves it. The Kuiva-side comparison
   (:func:`test_kuiva_soc_spectrum`) still skips, because a SOC state spectrum needs the conventional CI.
2. **The floor is 2%, not 0.5%.** Corrected splittings for *open-shell* atoms agree with
   four-component theory to 0.2-0.7% where closed shells reach 0.003%, and every target ion
   ion is open shell.
3. **A band set at the observed spread fails on the next system**, and the 11% already observed
   has a mechanism X2CAMF does not touch (see :data:`SPLITTING_RTOL`).

The tighter tier that *did* come out of X2CAMF is atomic and lives in
``tests/test_x2camf_dirac.py``, at 0.5% closed-shell and 2% open-shell against four-component
DIRAC. Different measurement, different thing measured.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pytest

from kuiva.props.multiplet import lande_g

REPO = Path(__file__).resolve().parents[1]
REF_MOLCAS = REPO / "tests/reference/tier2_molcas.json"
REF_DIRAC = REPO / "tests/reference/tier2_dirac.json"
#: Kuiva's own record, from ``tests/generate/tier2_kuiva.py``. Committed for the same reason
#: the external ones are: the sweep that produces it is hours (the four-component atomic solves).
REF_KUIVA = REPO / "tests/reference/tier2_kuiva.json"

# --- tolerances -------------------------------------------------------------------------
#: Lande g factors are analytic. 1% comfortably covers the small covalency/basis effects that
#: make a computed free-ion g deviate (observed: 0.04%), while catching a factor-of-two or a
#: g_e-vs-2 convention slip, which is the realistic failure mode.
G_LANDE_RTOL = 0.01
#: A free ion is spherically symmetric, so its three principal g values must coincide. This is
#: internal to one calculation, hence tight.
G_ISOTROPY_ATOL = 1e-3
#: Cross-code SOC splittings. The 4f splittings agree to ~5% (Ce(3+): 2428 vs 2299 cm^-1),
#: but the largest observed deviation is 11%, on Bi's 2P3/2 (OpenMolcas 32729, DIRAC 36390,
#: experiment 33165 cm^-1). That is understood rather than tolerated: DIRAC's reference is an
#: average-of-configuration DHF, whose orbitals are not relaxed for any individual state, so
#: the *highest* level of a minimal active space is pushed up, while OpenMolcas optimises
#: CASSCF orbitals per multiplicity and lands within 1.3% of experiment. 15% therefore admits
#: the known methodological spread without admitting a real error - the experimental anchor
#: below (30%) is what bounds gross mistakes.
SPLITTING_RTOL = 0.15
#: Degeneracy grouping; matches the generators.
DEGENERACY_TOL_CM = 1.0

# --- the Kuiva-side band (the standing Kuiva-side debt, now written) --------------------
#: Kuiva against OpenMolcas, in OpenMolcas's own basis, on SOC splittings.
#:
#: ⚠ **This is a different measurement from** :data:`SPLITTING_RTOL` **and it is the one that
#: measures Kuiva.** The 15% band above compares two *external* codes and nothing Kuiva does
#: moves it; this one compares Kuiva against one of them with the basis held fixed (the
#: matched-basis rule), so the residual is attributable to the Hamiltonian and the method.
#:
#: ⚠ **It is NOT the 2% of the atomic tier**, and the reason is worth stating because 2% is the
#: number anticipated. That tier (``tests/test_x2camf_dirac.py``) compares *the same
#: one-particle problem* solved two ways. Here **four** things differ at once: the scalar
#: Hamiltonian (X2C vs DKH2), the spin-orbit treatment (X2CAMF vs Breit-Pauli AMFI), the
#: orbitals (one two-component state-averaged set vs per-multiplicity RASSCF orbitals followed
#: by a RASSI mixing step) — and, until the matched-basis leg of ``tier2_kuiva.py`` exists, the
#: **basis**. A band set below the size of those fails on correct code, which is exactly what
#: a band must not do.
#:
#: ⚠ 20%, not the 15% the first two systems suggested. Observed: Bi +14.2%, TlH −13.7% — and
#: The standing warning is that *a band set at the observed spread is a band that fails on the
#: next system*. Setting it at the spread would have repeated the mistake this project already
#: recorded once. **Tighten it when the matched-basis leg removes the fourth confound**, not
#: before.
KUIVA_SPLITTING_RTOL = 0.20
#: Kuiva against **DIRAC**, the like-for-like comparison and the tighter of the two: DIRAC is
#: an X2C/four-component code, so it shares Kuiva's class of Hamiltonian where OpenMolcas's
#: DKH2+AMFI does not. Measured on Bi — Kuiva in ``x2c-SVPall-2c``, DIRAC in ``dyall.v2z``, so
#: this even absorbs a **basis** difference — the four levels agree to +2.75% … +3.44%, with a
#: spread between them of 0.7%: the shape of the spectrum is reproduced and what is left is an
#: almost uniform scale factor. 8% leaves room for the next system without admitting a real
#: error. ⚠ Tighten this only when the matched-basis leg of ``tier2_kuiva.py`` exists; until
#: then it is a cross-basis number and cannot be pushed to the method's real floor.
KUIVA_DIRAC_RTOL = 0.08
#: ⚠ **Absolute floor, and it is the load-bearing half of this tolerance for a ligand field.**
#: The largest *relative* deviations sit on the *smallest* splittings, which is the signature
#: that an absolute floor is the right instrument rather than a looser percentage. CeCl3's
#: first crystal-field level is the case: Kuiva 229.4 vs OpenMolcas 191.0 cm^-1 — 38 cm^-1
#: apart, which is +20.1% and would fail a 20% band while every larger level in the same
#: spectrum agrees to 0.3-9.6%. A crystal-field ladder is a set of small differences between
#: near-degenerate states and is the most basis- and orbital-sensitive thing either program
#: computes; 50 cm^-1 is a real disagreement at that scale and still two orders below any
#: *multiplet* splitting, so it cannot mask a method error where one would matter.
KUIVA_SPLITTING_ATOL_CM = 50.0
#: Free-ion g isotropy for Kuiva's own spectra. Looser than the external :data:`G_ISOTROPY_ATOL`
#: because Kuiva's orbitals are optimized in one two-component state-averaged calculation rather
#: than in a spherically symmetric atomic code, so the residual anisotropy is real numerical
#: symmetry breaking rather than zero by construction.
KUIVA_G_ISOTROPY_ATOL = 5e-3
#: Locality at 25 A: the residual interaction between two neutral, dipole-free CeCl3 units is
#: far below the meaningful scale for a magnetic spectrum.
ADDITIVITY_TOL_CM = 1.0


def _load(path: Path) -> Dict:
    if not path.is_file():
        pytest.skip(f"{path.relative_to(REPO)} missing; run its generator in tests/generate/")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def molcas() -> Dict:
    return _load(REF_MOLCAS)["records"]


@pytest.fixture(scope="module")
def dirac() -> Dict:
    return _load(REF_DIRAC)["records"]


def _ok(records: Dict, key: str) -> Dict:
    rec = records.get(key)
    if rec is None:
        pytest.skip(f"no record for {key}")
    if rec.get("status") != "ok":
        pytest.skip(f"{key}: reference not usable ({rec.get('status')})")
    return rec


# --- reference integrity -----------------------------------------------------------------
def test_molcas_reference_is_usable(molcas):
    assert molcas, "no OpenMolcas records"
    bad = {k: r.get("status") for k, r in molcas.items() if r.get("status") != "ok"}
    assert not bad, f"unusable OpenMolcas records: {bad}"


def test_dirac_reference_is_usable(dirac):
    assert dirac, "no DIRAC records"
    bad = {k: r.get("status") for k, r in dirac.items() if r.get("status") != "ok"}
    assert not bad, f"unusable DIRAC records: {bad}"


# --- degeneracy patterns: exact, symmetry-dictated ----------------------------------------
#: (key, pattern, why). These are the term symbols of the free-ion / ligand-field manifolds.
PATTERNS = [
    ("ce3p", [6, 8], "4f^1: 2F5/2 (6) below 2F7/2 (8)"),
    ("yb3p", [8, 6], "4f^13: the *inverted* multiplet, 2F7/2 (8) below 2F5/2 (6)"),
    ("bi", [4, 4, 6, 2, 4], "6p^3: 4S3/2, 2D3/2, 2D5/2, 2P1/2, 2P3/2"),
    ("cecl3", [2, 2, 2, 2, 2, 2, 2],
     "f^1 in a ligand field: every level is a Kramers doublet (7 of them)"),
    ("ticl3", [2, 2, 2, 2, 2],
     "d^1 in a D3h ligand field: every level is a Kramers doublet (5 of them). This is the "
     "single-site multiplet the ti2cl6_far spectrum must factorise into"),
]

#: Systems only OpenMolcas covers (no DiracSpec defined for the molecular cases).
_MOLCAS_ONLY = {"cecl3", "ticl3"}


@pytest.mark.parametrize("key,pattern,why", PATTERNS)
def test_molcas_degeneracy_pattern(molcas, key, pattern, why):
    assert _ok(molcas, key)["degeneracy_pattern"] == pattern, why


@pytest.mark.parametrize("key,pattern,why", [p for p in PATTERNS if p[0] not in _MOLCAS_ONLY])
def test_dirac_degeneracy_pattern(dirac, key, pattern, why):
    assert _ok(dirac, key)["degeneracy_pattern"] == pattern, why


def test_ce_and_yb_multiplets_are_inverted(molcas):
    """Less-than-half-filled (4f^1) puts J = L-S lowest; more-than-half-filled (4f^13) puts
    J = L+S lowest. Hund's third rule, and a check that the SOC operator carries the right
    sign - an error that leaves every splitting *magnitude* correct."""
    assert _ok(molcas, "ce3p")["degeneracy_pattern"] == [6, 8]
    assert _ok(molcas, "yb3p")["degeneracy_pattern"] == [8, 6]


# --- magnetic moments against analytic theory ---------------------------------------------
#: (key, L, S, multiplet index, description) for free ions whose g is analytic.
LANDE_CASES = [
    ("ce3p", 3, 0.5, 0, "Ce(3+) 2F5/2, g = 6/7"),
    ("ce3p", 3, 0.5, 1, "Ce(3+) 2F7/2, g = 8/7"),
    ("yb3p", 3, 0.5, 0, "Yb(3+) 2F7/2, g = 8/7"),
    ("yb3p", 3, 0.5, 1, "Yb(3+) 2F5/2, g = 6/7"),
    ("dy3p", 5, 2.5, 0, "Dy(3+) 6H15/2, g = 4/3 - the SMM ground multiplet"),
    ("dy3p", 5, 2.5, 1, "Dy(3+) 6H13/2"),
]


@pytest.mark.parametrize("key,l,s,index,desc", LANDE_CASES)
def test_free_ion_lande_g(molcas, key, l, s, index, desc):
    """The magnetic-moment matrices must reproduce the analytic Lande g factor.

    This is the single most valuable check in Tier 2: the target comes from angular-momentum
    theory, not from another program, so it validates the whole chain - SOC eigenstates,
    moment matrices, and the phase-invariant reduction of :mod:`kuiva.props.multiplet` -
    against something no implementation can argue with.
    """
    mult = _ok(molcas, key)["multiplets"][index]
    j = (mult["size"] - 1) / 2.0
    expected = lande_g(l, s, j)
    for g in mult["g_values"]:
        assert g == pytest.approx(expected, rel=G_LANDE_RTOL), \
            f"{desc}: got g = {mult['g_values']}, expected {expected:.6f}"


@pytest.mark.parametrize("key", ["ce3p", "yb3p", "dy3p"])
def test_free_ion_g_tensor_is_isotropic(molcas, key):
    """A free ion has no preferred direction, so the three principal g values must coincide.

    Anisotropy here would mean the moment matrices had picked up a spurious axis - e.g. from
    a symmetry-broken orbital set - which is easy to miss when only g_z is inspected.
    """
    for mult in _ok(molcas, key)["multiplets"][:3]:
        g = mult["g_values"]
        assert max(g) - min(g) < G_ISOTROPY_ATOL, f"{key}: anisotropic free-ion g = {g}"


def test_ligand_field_g_is_anisotropic(molcas):
    """CeCl3, by contrast, *must* be anisotropic: the ligand field defines an axis.

    Paired with the isotropy test above, this shows the moment machinery is sensitive to real
    anisotropy rather than trivially returning isotropic values.
    """
    g = _ok(molcas, "cecl3")["multiplets"][0]["g_values"]
    assert max(g) - min(g) > 0.5, f"expected an anisotropic ground doublet, got {g}"


# --- cross-code consistency ----------------------------------------------------------------
@pytest.mark.parametrize("key", ["ce3p", "yb3p", "bi"])
def test_molcas_and_dirac_agree_on_structure(molcas, dirac, key):
    """Two independent programs, two different relativistic treatments, same structure.

    Degeneracies must match exactly; the splittings only to ~10%, which is the honest size of
    the DKH2+AMFI vs X2C difference. Tightening this would make the suite fail on correct
    code - the point of the test is that both codes describe the *same states*.
    """
    m, d = _ok(molcas, key), _ok(dirac, key)
    assert m["degeneracy_pattern"] == d["degeneracy_pattern"], \
        f"{key}: OpenMolcas {m['degeneracy_pattern']} vs DIRAC {d['degeneracy_pattern']}"
    m_e = [x["energy_cm"] for x in m["multiplets"]]
    d_e = [x["energy_cm"] for x in d["multiplets"]]
    for i, (a, b) in enumerate(zip(m_e[1:], d_e[1:]), start=1):
        assert a == pytest.approx(b, rel=SPLITTING_RTOL), \
            f"{key} level {i}: OpenMolcas {a:.1f} vs DIRAC {b:.1f} cm-1"


#: Experimental free-ion levels [cm^-1] (NIST Atomic Spectra Database; Ce(3+)/Yb(3+) 4f
#: multiplet splittings from optical spectroscopy). These are *not* pass/fail targets - a
#: CASSCF-level calculation in a DZ basis is not expected to reproduce them - but they anchor
#: the reference data to physical reality and would expose an error of the wrong order.
EXPERIMENT_CM = {
    "ce3p": [0.0, 2253.0],
    "yb3p": [0.0, 10214.0],
    "bi": [0.0, 11419.0, 15438.0, 21661.0, 33165.0],
}


@pytest.mark.parametrize("key", sorted(EXPERIMENT_CM))
def test_splittings_are_physically_reasonable(molcas, key):
    """Computed SOC splittings must be within 30% of experiment.

    Deliberately loose: this is a sanity anchor against a mis-specified active space or a
    spin-orbit term that is off by a large factor, not an accuracy assessment.
    """
    got = [m["energy_cm"] for m in _ok(molcas, key)["multiplets"]]
    exp = EXPERIMENT_CM[key]
    assert len(got) == len(exp), f"{key}: {len(got)} levels, expected {len(exp)}"
    for i, (g, e) in enumerate(zip(got[1:], exp[1:]), start=1):
        assert abs(g - e) / e < 0.30, \
            f"{key} level {i}: {g:.0f} cm-1 vs experimental {e:.0f} cm-1"


# --- the tiers meet: scalar agreement isolates the SOC treatment ---------------------------
def test_scalar_casscf_agrees_across_codes():
    """PySCF's scalar CASSCF and OpenMolcas's *spin-free* RASSCF must agree closely.

    Both are run in the same basis (``ano-rcc-vdzp``) on the same CeCl3 geometry and active
    space, differing only in the scalar relativistic Hamiltonian (X2C vs DKH2). Observed
    agreement on the 4f crystal-field splitting is ~1 cm^-1 out of 1300 (0.05%).

    This is what licenses the loose Tier-2 tolerances: since the *scalar* problem is
    demonstrably the same in both codes to within a cm^-1, any Tier-2 discrepancy of hundreds
    of cm^-1 is attributable to the spin-orbit treatment rather than to a mismatched setup.
    Without this bridge, a wrong active space and a wrong SOC operator look alike.
    """
    tier1_path = REPO / "tests/reference/tier1_pyscf.json"
    if not tier1_path.is_file():
        pytest.skip("tier1_pyscf.json missing")
    t1 = json.loads(tier1_path.read_text())["records"].get("cecl3/ano-rcc-vdzp")
    if t1 is None or "error" in t1:
        pytest.skip("no scalar CeCl3 record (it is a slow system to generate)")
    t2 = _ok(_load(REF_MOLCAS)["records"], "cecl3")
    if "spinfree_rel_cm" not in t2:
        pytest.skip("no spin-free energies stored")

    pyscf_levels = sorted(t1["e_states_rel_cm"])
    molcas_levels = sorted(t2["spinfree_rel_cm"])
    assert len(pyscf_levels) == len(molcas_levels) == 7
    for i, (a, b) in enumerate(zip(pyscf_levels, molcas_levels)):
        assert a == pytest.approx(b, abs=5.0), \
            f"scalar level {i}: PySCF {a:.1f} vs OpenMolcas {b:.1f} cm-1"


# --- multi-site: the tensor-product structure ----------------------------------------------
def _blocks(rec) -> List[Dict]:
    return rec["multiplets"]


def _reconstruct_site_levels(levels_cm: List[float], n_sites_levels: int,
                             tol: float) -> List[float]:
    """Recover the single-site spectrum from a two-site spectrum of pairwise sums.

    If the two sites are independent, the dimer levels are exactly ``{m_i + m_j}``. Working up
    from the ground state, the lowest not-yet-explained level must be ``m_0 + m_k`` for the
    next unknown ``m_k``; adding it generates further sums. Greedy reconstruction is
    unambiguous here because ``m_0 = 0``.
    """
    distinct: List[float] = []
    for e in sorted(levels_cm):
        if not distinct or e - distinct[-1] > tol:
            distinct.append(e)
    site = [0.0]
    generated = {0.0}
    for e in distinct:
        if any(abs(e - g) <= tol for g in generated):
            continue
        site.append(e)
        generated = {a + b for a in site for b in site}
        if len(site) == n_sites_levels:
            break
    return site


def test_site_reconstruction_recovers_known_levels():
    """Verify the reconstruction helper itself on a synthetic two-site spectrum.

    The multi-site test below is only as trustworthy as this greedy inversion, so it is
    checked against a case whose answer is known by construction - including the interleaving
    that makes the problem non-trivial (here 2*m1 = 382 falls *below* m2 = 1022, so the
    dimer levels are not simply the site levels followed by sums).
    """
    site = [0.0, 191.0, 1022.0, 2347.0, 2481.0, 2815.0, 3598.0]
    dimer = sorted({round(a + b, 6) for a in site for b in site})
    assert _reconstruct_site_levels(dimer, 7, ADDITIVITY_TOL_CM) == pytest.approx(site)


def test_far_dimer_factorises_into_local_multiplets(molcas):
    """**The multi-site test.** At 25 A the two d^1 centres cannot interact, so the 100-state
    spectrum must be exactly the tensor product of two 10-state local multiplets: every level
    is ``E_A + E_B`` and every degeneracy is a product of local degeneracies.

    This is the sharpest available check of the local-multiplet picture that
    ``dmrg/manifold.py`` is built on, and of size consistency - and it costs no more
    than the coupled dimer. If a future multi-site implementation gets local multiplets wrong,
    it fails here first.
    """
    rec = _ok(molcas, "ti2cl6_far")
    assert rec["n_soc_states"] == 100, "expected 10 x 10 local-multiplet product states"

    blocks = _blocks(rec)
    # Each level is a product of two Kramers doublets: 2 x 2 = 4, doubled to 8 when the two
    # sites occupy *different* local levels (E_i + E_j == E_j + E_i).
    for b in blocks:
        assert b["size"] % 4 == 0, \
            f"block of {b['size']} states is not a product of two Kramers doublets"

    site = _reconstruct_site_levels([b["energy_cm"] for b in blocks], 5, ADDITIVITY_TOL_CM)
    assert len(site) == 5, f"expected 5 local levels, reconstructed {len(site)}: {site}"

    # Every observed level must be a sum of two reconstructed local levels.
    predicted = sorted({round(a + b, 6) for a in site for b in site})
    for b in blocks:
        assert any(abs(b["energy_cm"] - p) <= ADDITIVITY_TOL_CM for p in predicted), \
            f"level {b['energy_cm']:.2f} cm-1 is not a sum of local levels {site}"


def test_far_dimer_local_levels_match_the_monomer(molcas):
    """The local levels recovered from the dimer must be the TiCl3 monomer's own spectrum.

    The far dimer is built from two copies of exactly the monomer geometry, so this closes the
    loop: a 100-state two-site calculation is inverted back to the 5-level single-site
    spectrum it is made of. Only the state-averaging differs (5 roots vs 25), which is worth
    about 0.1 cm^-1 in practice; 5 cm^-1 leaves room for that without admitting a real error.
    """
    mono = _ok(molcas, "ticl3")
    dimer = _ok(molcas, "ti2cl6_far")
    site = _reconstruct_site_levels([b["energy_cm"] for b in _blocks(dimer)], 5,
                                    ADDITIVITY_TOL_CM)
    mono_levels = [b["energy_cm"] for b in _blocks(mono)]
    assert len(site) == len(mono_levels) == 5
    for a, b in zip(site, mono_levels):
        assert a == pytest.approx(b, abs=5.0), \
            f"local levels {site} vs monomer {mono_levels}"


def test_coupled_dimer_has_the_same_manifold_as_the_far_dimer(molcas):
    """Bringing the two centres together must not change *how many* states there are.

    The 100 states of the d^1 (x) d^1 manifold are fixed by the active space; the interaction
    only redistributes them in energy. A different count would mean the coupled calculation
    is describing a different space (e.g. charge-transfer configurations leaking in), which
    would invalidate the comparison between the two geometries.
    """
    coupled = _ok(molcas, "ti2cl6")
    far = _ok(molcas, "ti2cl6_far")
    assert coupled["n_soc_states"] == far["n_soc_states"] == 100


def test_coupling_does_not_increase_degeneracy(molcas):
    """Interaction can only split degenerate levels, never create new degeneracy.

    Ti(III)-Ti(III) exchange through the chloride bridge is small, so this asserts the
    direction of the effect rather than a magnitude: the coupled dimer's ground block must
    not be *more* degenerate than the non-interacting one.
    """
    coupled = _ok(molcas, "ti2cl6")
    far = _ok(molcas, "ti2cl6_far")
    assert _blocks(coupled)[0]["size"] <= _blocks(far)[0]["size"]


# --- Kuiva reproduces the reference ---------------------------------------------------------
#
# ⚠ These read Kuiva's **committed** record (``tier2_kuiva.json``, from
# ``tests/generate/tier2_kuiva.py``) rather than running the calculation, for the same reason
# the external records are committed: the sweep that produces it is hours, dominated by the
# four-component atomic solves. What keeps that honest is
# :func:`test_kuiva_record_still_reproduces`, which re-runs the cheapest system end to end and
# is marked ``slow``; without it the record could rot into a self-consistent fiction.
#
# ⚠ **Everything below compares phase-invariant reductions only**: degeneracy patterns,
# relative energies, and the principal g values of ``M_ij = Tr_block(mu_i mu_j)``. Never a
# matrix element — the dump fixes no phase convention.


def _systems():
    """``tests/generate/systems.py`` — the single source of truth for what a system *is*."""
    import sys

    path = str(REPO / "tests/generate")
    if path not in sys.path:
        sys.path.insert(0, path)
    import systems                                                      # noqa: PLC0415
    return systems


@pytest.fixture(scope="module")
def kuiva() -> Dict:
    return _load(REF_KUIVA)["records"]


def _kuiva_record(records: Dict, key: str, basis: str = "matched") -> Dict:
    """The record for ``key``, preferring the basis the external code used.

    Matched-basis first, because a cross-code comparison is only attributable to the
    Hamiltonian if the basis is held fixed; the project-default record is the fallback and is
    reported as such when it is the one used.
    """
    system = _systems().SYSTEMS_BY_KEY[key]
    order = ([system.basis_matched, system.basis] if basis == "matched"
             else [system.basis, system.basis_matched])
    for name in order:
        rec = records.get("{}/{}".format(key, name))
        if rec is not None and rec.get("status") == "ok":
            return rec
    pytest.skip("no usable Kuiva record for {} (have {})".format(
        key, sorted(k for k in records if k.startswith(key + "/"))))


def test_kuiva_reference_is_usable(kuiva):
    assert kuiva, "no Kuiva records"
    bad = {k: r.get("status") for k, r in kuiva.items() if r.get("status") != "ok"}
    assert not bad, "unusable Kuiva records: {}".format(bad)


def test_kuiva_records_say_which_hamiltonian_produced_them(kuiva):
    """⚠ the ingestion rule's contract, enforced where it matters: a stored spin-orbit result that does not
    say whether the two-electron picture change was included is not interpretable, and the
    difference is 5-30% on every splitting in it. An *unscreened* record silently compared
    against these tolerances is the failure mode this test exists for."""
    for tag, rec in kuiva.items():
        if rec.get("status") != "ok":
            continue
        provenance = rec.get("hamiltonian", {})
        assert provenance, "{}: no Hamiltonian provenance stored".format(tag)
        assert provenance["screening"]["method"] == "x2camf", \
            "{}: screening = {!r}; a Tier-2 record must carry the two-electron picture " \
            "change ".format(tag, provenance["screening"]["method"])


def test_kuiva_casscf_converged(kuiva):
    """A record from an unconverged orbital optimization is not a measurement of the method."""
    stalled = {tag: rec["casscf_grad_norm"] for tag, rec in kuiva.items()
               if rec.get("status") == "ok" and not rec.get("casscf_converged", False)}
    assert not stalled, "CASSCF did not converge for: {}".format(stalled)


@pytest.mark.parametrize("key,pattern,why", PATTERNS)
def test_kuiva_degeneracy_pattern(kuiva, key, pattern, why):
    """**Exact**, in both bases where both exist: degeneracies come from symmetry and from
    Kramers' theorem, and no method, basis or screening choice can change them."""
    assert _kuiva_record(kuiva, key)["degeneracy_pattern"] == pattern, why


@pytest.mark.parametrize("key,l,s,index,desc", LANDE_CASES)
def test_kuiva_free_ion_lande_g(kuiva, key, l, s, index, desc):
    """⚠ **The sharpest check in the suite, now applied to Kuiva itself.**

    The target is angular-momentum theory, not another program, so this validates the whole
    chain at once — the X2C spin-orbit operator with its X2CAMF screening, the two-component
    CASSCF, the CI states, the transition densities, ``L`` and ``S``, and the phase-invariant
    reduction — against something no implementation can argue with.
    """
    mult = _kuiva_record(kuiva, key)["multiplets"][index]
    j = (mult["size"] - 1) / 2.0
    expected = lande_g(l, s, j)
    for g in mult["g_values"]:
        assert g == pytest.approx(expected, rel=G_LANDE_RTOL), \
            "{}: got g = {}, expected {:.6f}".format(desc, mult["g_values"], expected)


@pytest.mark.parametrize("key", ["ce3p", "yb3p", "dy3p"])
def test_kuiva_free_ion_g_tensor_is_isotropic(kuiva, key):
    """A free ion has no preferred direction. Anisotropy here would mean the moment matrices
    had picked up a spurious axis — from a symmetry-broken orbital set, say — which is easy to
    miss when only one principal value is inspected."""
    for mult in _kuiva_record(kuiva, key)["multiplets"][:3]:
        g = mult["g_values"]
        assert max(g) - min(g) < KUIVA_G_ISOTROPY_ATOL, \
            "{}: anisotropic free-ion g = {}".format(key, g)


def test_kuiva_ligand_field_g_is_anisotropic(kuiva):
    """CeCl3 must be anisotropic: the ligand field defines an axis. Paired with the isotropy
    test above, this shows the moment machinery responds to real anisotropy rather than
    trivially returning a plausible constant."""
    g = _kuiva_record(kuiva, "cecl3")["multiplets"][0]["g_values"]
    assert max(g) - min(g) > 0.5, "expected an anisotropic ground doublet, got {}".format(g)


@pytest.mark.parametrize("key", [p[0] for p in PATTERNS])
def test_kuiva_and_molcas_agree_on_splittings(kuiva, molcas, key):
    """Kuiva against OpenMolcas, in OpenMolcas's own basis, through the invariants only.

    ⚠ **This band is not the 15% of** :data:`SPLITTING_RTOL`. That one compares two *external*
    codes and does not measure Kuiva at all. This one does, in a matched basis, and
    :data:`KUIVA_SPLITTING_RTOL` records what it is worth — see its comment for why the number
    is what it is rather than the 2% an atomic comparison reaches.
    """
    ours = _kuiva_record(kuiva, key)
    theirs = _ok(molcas, key)
    assert ours["degeneracy_pattern"] == theirs["degeneracy_pattern"]
    same_basis = ours["basis"] == _systems().get(key).basis_matched
    mine = [m["energy_cm"] for m in ours["multiplets"]]
    yours = [m["energy_cm"] for m in theirs["multiplets"]]
    for i, (a, b) in enumerate(zip(mine[1:], yours[1:]), start=1):
        assert a == pytest.approx(b, rel=KUIVA_SPLITTING_RTOL, abs=KUIVA_SPLITTING_ATOL_CM), \
            "{} level {}: Kuiva {:.1f} ({}) vs OpenMolcas {:.1f} ({}){}".format(
                key, i, a, ours["basis"], b, theirs.get("basis"),
                "" if same_basis else "  [cross-basis: the band absorbs a basis difference "
                                      "too, see KUIVA_SPLITTING_RTOL]")


@pytest.mark.parametrize("key", ["ce3p", "yb3p", "bi"])
def test_kuiva_and_dirac_agree_on_splittings(kuiva, dirac, key):
    """⚠ **The like-for-like comparison, and the tighter of the two.**

    DIRAC is the X2C/four-component code, so it shares Kuiva's *class* of Hamiltonian in a way
    OpenMolcas's DKH2+AMFI does not. Measured on Bi across three different bases (Kuiva
    ``x2c-SVPall-2c``, DIRAC ``dyall.v2z``), the four levels agree to **+2.75% to +3.44%** —
    and the spread *between* those four is 0.7%, i.e. Kuiva reproduces the shape of the
    spectrum and differs by an almost uniform scale factor. Against OpenMolcas the same
    numbers are +6.4% to +14.2%.

    That is why there are two bands here rather than one: they measure different things, and
    collapsing them would throw away the more informative comparison.
    """
    ours = _kuiva_record(kuiva, key)
    theirs = _ok(dirac, key)
    assert ours["degeneracy_pattern"] == theirs["degeneracy_pattern"]
    mine = [m["energy_cm"] for m in ours["multiplets"]]
    yours = [m["energy_cm"] for m in theirs["multiplets"]]
    for i, (a, b) in enumerate(zip(mine[1:], yours[1:]), start=1):
        assert a == pytest.approx(b, rel=KUIVA_DIRAC_RTOL, abs=KUIVA_SPLITTING_ATOL_CM), \
            "{} level {}: Kuiva {:.1f} vs DIRAC {:.1f} cm-1".format(key, i, a, b)


def test_kuiva_far_dimer_factorises_into_local_multiplets(kuiva):
    """The multi-site test, on Kuiva's own spectrum: at 25 A the 100 states must be exactly
    the tensor product of two 10-state local multiplets, and the reconstructed local levels
    must be the TiCl3 monomer's own."""
    rec = _kuiva_record(kuiva, "ti2cl6_far", basis="project")
    blocks = _blocks(rec)
    assert sum(b["size"] for b in blocks) == 100
    for b in blocks:
        assert b["size"] % 4 == 0, \
            "block of {} states is not a product of two Kramers doublets".format(b["size"])
    site = _reconstruct_site_levels([b["energy_cm"] for b in blocks], 5, ADDITIVITY_TOL_CM)
    assert len(site) == 5, "expected 5 local levels, reconstructed {}: {}".format(
        len(site), site)
    mono = [b["energy_cm"] for b in _blocks(_kuiva_record(kuiva, "ticl3", basis="project"))]
    for a, b in zip(site, mono):
        assert a == pytest.approx(b, abs=5.0), \
            "local levels {} vs monomer {}".format(site, mono)


#: ⚠ **A free ion's ``2J+1`` multiplets are degenerate by spherical symmetry — exactly.**
#: A splitting of 0.1 cm^-1 already implies different physics, so this is not a numerical
#: tolerance to be set at whatever the code happens to produce; it is a physical statement.
#: 0.05 cm^-1 keeps a factor of two below that scale while sitting an order above what the
#: code now delivers. Measured at the current ``DEFAULT_CHOLESKY_TOL`` (1e-8) with the
#: orbit-complete pivoting: Bi 0.00009, Yb(3+) 0.00005, Ce(3+) 0.0066 cm^-1 — all with
#: >=7x margin. ⚠ Deliberately **not** set at the observed 0.0066: the rule is that a
#: band set at the observed spread fails on the next system, and the number that matters is the
#: physical one.
#:
#: ⚠ **Ce(3+) is the outlier and it is a lead, not noise.** Pivoting on symmetry orbits moved
#: Bi 2.6x and Yb(3+) **43x** (0.00215 -> 0.00005) and moved Ce(3+) not at all (0.00645 ->
#: 0.00662) — same shell as Yb(3+), same protocol, one *electron* against one *hole*, and 140x
#: apart. Whatever is left in Ce(3+) is not the factorization — it is **convergence**: the
#: spread tracks the orbital gradient norm almost linearly (8.41 cm^-1 at |g| = 2.0e-1, 2.02 at
#: 6.2e-2, 0.53 at 1.2e-2, 0.0066 at 6.7e-4), and Ce(3+)'s averaged density is exactly
#: spherical throughout (all 14 occupations 1/14 to 3e-16). Yb(3+) simply converges further.
#:
#: ⚠ **This does not apply to a complex.** In CeCl3 or TiCl3 the ligand field genuinely splits
#: the free-ion multiplets, and what must then be right is the *pattern* — the point-group
#: degeneracies, which :data:`PATTERNS` asserts (every level a Kramers doublet for an
#: odd-electron ion in D3h).
FREE_ION_DEGENERACY_TOL_CM = 0.05


def _is_free_ion(key: str) -> bool:
    return len(_systems().get(key).atoms) == 1


@pytest.mark.parametrize("key", ["ce3p", "yb3p", "bi", "dy3p"])
def test_kuiva_free_ion_multiplets_are_degenerate(kuiva, key):
    """⚠ **The guard for a defect that produced entirely plausible numbers.**

    A pivoted Cholesky decomposition is not rotationally invariant, so its truncation error
    splits degeneracies that symmetry makes exact. At the old 1e-6 threshold this gave
    Ce(3+) 0.22 cm^-1 and Dy(3+) **44.85 cm^-1** — the latter large enough to shatter the
    16-fold ⁶H15/2 ground multiplet into eight doublets and turn its recorded Landé g into a
    *doublet's* g. Nothing about those records looked wrong.

    ``tests/test_integrals.py`` guards the mechanism directly and cheaply; this guards the
    observable, on the systems the program exists to compute.

    ⚠ **Dy(3+) was a `strict=True` xfail here for exactly one reason and it is now fixed.**
    The cause was never the integrals: it was the *state average*. 126 is the spin-free sextet
    count and is **not** a spinor manifold boundary — roots 119-134 are one 16-fold manifold,
    so a 126-state average cut it in half, the averaged density stopped being spherical, and
    the ground manifold split by 44.85 cm^-1. At 134 the worst spread is 0.005 cm^-1. See
    :func:`test_kuiva_state_average_boundary_is_clean`, which guards the *mechanism* the way
    this guards the observable.
    """
    assert _is_free_ion(key), "{} is not a free ion; see FREE_ION_DEGENERACY_TOL_CM".format(key)
    record = _kuiva_record(kuiva, key)
    worst = max((m["spread_cm"], m["size"], m["energy_cm"]) for m in record["multiplets"])
    assert worst[0] < FREE_ION_DEGENERACY_TOL_CM, (
        "{}: a {}-fold multiplet at {:.1f} cm^-1 is split by {:.4f} cm^-1, which spherical "
        "symmetry makes exactly zero. Above ~0.1 cm^-1 this is different physics, not noise."
        .format(key, worst[1], worst[2], worst[0]))


#: Dy(3+) 4f^9: the ⁶H and ⁶F ``2J+1`` ladders, from angular-momentum theory alone.
#: ⁶H (L=5, S=5/2) -> J = 15/2..5/2 -> 16,14,12,10,8,6;
#: ⁶F (L=3, S=5/2) -> J = 11/2..1/2 -> 12,10,8,6,4,2.  108 states, and they interleave.
#:
#: ⚠ **⁶P is deliberately NOT in this list, and that is a measurement rather than an
#: omission.** The spin-free sextet count says the next 18 states are ⁶P (8, 6, 4). Kuiva's
#: two-component spectrum instead shows a **10**-fold manifold at 24.3 kcm^-1 and a **16**-fold
#: at 25.0 kcm^-1, each degenerate to 0.003 cm^-1 — i.e. the ⁶P region is thoroughly mixed with
#: quartets that SOC brings down to meet it, and "the 126 sextet states" is not a set the
#: spinor spectrum contains. That is the same fact that made 126 the wrong state count, seen
#: from the other side; see ``systems.dy3p``.
DY_LOWER_LADDER = sorted([16, 14, 12, 10, 8, 6] + [12, 10, 8, 6, 4, 2])


def test_kuiva_dy_multiplet_decomposition(kuiva):
    """The f^9 decomposition against angular-momentum theory.

    ⚠ Asserted as a **multiset plus the leading ladder**, not as an ordered pattern: the ⁶H
    ladder (16, 14, 12, 10) is unambiguous and must come first, but two manifolds near
    9.5 kcm^-1 are only 168 cm^-1 apart in OpenMolcas and their order is genuinely
    method-dependent. Demanding the full order would fail on correct code.
    """
    pattern = _kuiva_record(kuiva, "dy3p")["degeneracy_pattern"]
    assert sum(pattern) == 134, "the dy3p record averages over 134 spinor roots"
    assert pattern[:4] == [16, 14, 12, 10], \
        "the ⁶H ladder must be lowest and in order; got {}".format(pattern[:4])


def test_kuiva_dy_full_term_decomposition(kuiva):
    """The ⁶H/⁶F part of the f^9 decomposition, as a multiset.

    ⚠ This test was an ``xfail`` until the state count was fixed, and it could not have been
    anything else: at 126 roots the 44.85 cm^-1 intra-manifold spread forced a 100 cm^-1
    grouping tolerance, which merged manifolds closer than that and made the ladder
    unresolvable. At 134 roots every manifold is degenerate to 0.005 cm^-1 and the default
    1 cm^-1 grouping resolves all of them.
    """
    pattern = _kuiva_record(kuiva, "dy3p")["degeneracy_pattern"]
    lower = sorted(pattern[:12])
    assert lower == DY_LOWER_LADDER, (
        "the lowest 108 f^9 states must decompose into the ⁶H/⁶F J-ladders {}; got {}"
        .format(DY_LOWER_LADDER, lower))
    assert sum(pattern[:12]) == 108


#: A state-averaged CASSCF is only as symmetric as the set it averages over, so the average
#: must not end *inside* a near-degenerate manifold. ``boundary_gap_cm`` is the distance from
#: the last averaged root to the first one left out, measured by a CASCI at the converged
#: orbitals (``tier2_kuiva.BOUNDARY_MARGIN``).
#:
#: ⚠ 50 cm^-1 is not a tolerance on a physical quantity — it is a statement that the boundary
#: is *unambiguous*. Measured: Dy(3+) at the fixed 134 roots has **2058 cm^-1**, and at the old
#: 126 it had **3.9**, while the manifold it cut spanned 24.5. Anything of that order means the
#: selection can reorder across the boundary during the optimization, which is precisely how
#: the C1 broken-average defect arose.
BOUNDARY_GAP_MIN_CM = 50.0


def test_kuiva_state_average_boundary_is_clean(kuiva):
    """⚠ **The guard for C1's actual mechanism, as opposed to its observable.**

    A count that cuts a manifold makes the averaged density non-invariant, the Fock operator
    built from it splits the shell, and the orbitals never recover — self-consistently, with
    every number in the record still plausible. Nothing else in the suite can see it: the
    energies, the g values and the moment matrices of the cut calculation all looked fine.

    ⚠ **A ``null`` gap is a pass, not a gap of zero**, and the distinction is the physics: Bi,
    Ce(3+) and Yb(3+) average over their *entire* CI space (20 states of 20 determinants, 14 of
    14), so no root is left out and the average is complete by construction. Records written
    before this field existed are skipped rather than passed — a missing measurement is not a
    clean boundary.
    """
    checked = []
    for tag, rec in kuiva.items():
        if rec.get("status") != "ok" or "boundary_gap_cm" not in rec:
            continue
        checked.append(tag)
        if rec["boundary_gap_cm"] is None:
            assert rec.get("boundary_spans_full_ci"), (
                "{}: boundary_gap_cm is null but the average does not span the full CI space; "
                "a missing measurement must not read as a clean boundary".format(tag))
            continue
        assert rec["boundary_gap_cm"] > BOUNDARY_GAP_MIN_CM, (
            "{}: the state average ends {:.2f} cm^-1 below the first root it leaves out, so "
            "the {}-state set may be cutting a manifold. Next energies: {}"
            .format(tag, rec["boundary_gap_cm"], rec["n_soc_states"],
                    rec.get("boundary_next_cm")))
    if not checked:
        pytest.skip("no record carries boundary_gap_cm yet")


def test_kuiva_moment_matrices_are_hermitian(kuiva):
    """A structural invariant of the dump, checked on every stored record at once."""
    for tag, rec in kuiva.items():
        if rec.get("status") == "ok":
            assert rec["moment_hermiticity"] < 1e-10, tag


def test_kuiva_inactive_space_carries_no_moment(kuiva):
    """`L` and `S` are time odd, so a Kramers-paired inactive space contributes exactly zero
. Nonzero here would mean the CASSCF had broken Kramers symmetry in the core."""
    for tag, rec in kuiva.items():
        if rec.get("status") == "ok":
            assert max(abs(x) for x in rec["inactive_moment_l"]) < 1e-8, tag


@pytest.mark.slow
def test_kuiva_record_still_reproduces():
    """⚠ Re-run the cheapest Tier-2 system end to end and check the committed record.

    Without this the Kuiva record is a self-consistent fiction: every test above would keep
    passing on a stored file after the code that produced it had broken. TiCl3 is chosen
    because it is the cheapest system that exercises the whole chain — front-end, X2CAMF,
    SA-CASSCF, moment matrices, phase-invariant reduction — and because its degeneracy pattern
    is fixed by symmetry, so a regression cannot hide behind a plausible number.
    """
    sysdef = _systems()
    import tier2_kuiva                                                  # noqa: PLC0415

    records = _load(REF_KUIVA)["records"]
    system = sysdef.SYSTEMS_BY_KEY["ticl3"]
    tag = "{}/{}".format(system.key, system.basis)
    stored = records.get(tag)
    if stored is None or stored.get("status") != "ok":
        pytest.skip("no stored Kuiva record for {}".format(tag))

    fresh = tier2_kuiva.run_system(system, system.basis, screening=stored["screening"],
                                   memory_gb=8.0, max_iter=tier2_kuiva.MAX_ITER)
    assert fresh["degeneracy_pattern"] == stored["degeneracy_pattern"]
    assert fresh["e_casscf_avg"] == pytest.approx(stored["e_casscf_avg"], abs=1e-7)
    for a, b in zip(fresh["multiplets"], stored["multiplets"]):
        assert a["energy_cm"] == pytest.approx(b["energy_cm"], abs=1.0)
        assert a["g_values"] == pytest.approx(b["g_values"], abs=1e-3)


def test_kuiva_multisite_local_multiplets():
    """Kuiva's multi-site manifold machinery against the ti2cl6_far factorisation.

    ⚠ This ran, and its numbers are recorded: the far
    dimer's 100-state manifold factorizes to 0.0011 cm^-1 against the same-integral CI.
    It stays skipped here because it is a multi-hour ab initio run, not a suite test —
    ``tests/generate/manifold_ladder.py`` is its generator and
    ``tests/reference/manifold_ladder.json`` its record.
    """
    pytest.importorskip("kuiva.dmrg.manifold")
    pytest.skip("multi-hour generator run; see tests/generate/manifold_ladder.py")
