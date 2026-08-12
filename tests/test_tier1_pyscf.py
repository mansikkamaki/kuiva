"""Tier-1 tests: scalar references from PySCF.

Three kinds of test live here, and the distinction matters:

1. **Bridge regression** — Kuiva's front-end is re-run and must reproduce the stored
   scalar X2C SCF energy to 1e-8 Eh. This exercises real Kuiva code *today*.
2. **Reference sanity** — assertions on the stored data itself: degeneracy patterns fixed by
   symmetry, the CASCI-equals-SCF invariance, NEVPT2 degeneracy. These guard against a bad
   regeneration silently replacing good reference values with plausible-looking wrong ones,
   which is the main failure mode of a committed-reference scheme.
3. **Method placeholders** — the comparisons Kuiva's CI/CASSCF/NEVPT2 must pass once those
   modules exist. They skip themselves until the module is importable, so they light up
   automatically rather than needing to be remembered.

Every number carries an explicit tolerance and a note on the
*physically* meaningful scale.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pytest

REPO = Path(__file__).resolve().parents[1]
REF = REPO / "tests/reference/tier1_pyscf.json"

# --- tolerances -------------------------------------------------------------------------
# ⚠ **A TOTAL energy is locked RELATIVELY; a SPLITTING is locked ABSOLUTELY** (user decision).
# The two are different quantities and the distinction is the whole content of this block.
#
# A total energy carries the whole core, so its magnitude is an accident of the element:
# -128 Eh for Ne, -8855 for Ce(3+), -21532 for Bi. An *absolute* lock on it therefore demands
# 168x more relative precision from Bi than from Ne for no physical reason, and it is not
# portable across a toolchain: the oneAPI 2023.2 -> 2026.1 update moved these stored energies
# by 3.2e-12 (Bi SCF), 1.8e-10 (Ce(3+) SCF) and 1.1e-10 (Ce(3+) CASSCF) **relative**, which
# broke three 1e-8/1e-7 Eh absolute locks while being, in any physical sense, nothing at all.
# A small systematic shift in an absolute energy is not dangerous; what is dangerous is a
# shift in an energy *difference*, and those are locked absolutely elsewhere — `DEGENERACY_CM`
# below, the 0.05 cm^-1 free-ion multiplet band, and `rdm.DEFAULT_DEGENERACY_TOL`.
#
# Margins over the measured drift: 5.5x (SCF) and 9x (CASSCF). Light systems agree far better
# than this — ~7e-12 relative on `ne`/`zn2p` — so these bands are not tight for them, which is
# the deliberate cost of one band that means the same thing for every element.
#: Stored-reference regression lock on a total SCF energy. Same code and same inputs, but
#: **not necessarily the same toolchain** as the record, which is what makes it relative.
E_SCF_RTOL = 1e-9
#: Kuiva's CASSCF against the stored PySCF value: a *cross-implementation* comparison, so it
#: gets an order more room than the regression lock, exactly as the absolute pair did.
E_CAS_RTOL = 1e-8
#: ⚠ Stays **absolute**, and is a different thing from the two above: it bounds comparisons
#: between two energies produced by the *same* run (a variational bound, and the one-
#: determinant CASSCF-equals-SCF identity). No toolchain drift can enter between them, so
#: there is nothing for a relative band to buy.
E_CAS_TOL = 1e-7
#: Degeneracies dictated by spherical symmetry. A converged CI holds these to ~1e-3 cm^-1;
#: 1 cm^-1 is a generous ceiling that still catches any real symmetry breaking.
DEGENERACY_TOL_CM = 1.0
#: SC-NEVPT2 corrections within a degenerate manifold. Looser than one might expect: PySCF's
#: degenerate CASCI roots mix arbitrarily, and the strongly contracted correction then varies
#: at the 1e-4 Eh level. That is a property of the reference data, not a physical splitting.
E_NEVPT2_DEGEN_TOL = 5e-4


@pytest.fixture(scope="module")
def ref() -> Dict:
    if not REF.is_file():
        pytest.skip(f"{REF.relative_to(REPO)} missing; run tests/generate/tier1_pyscf.py")
    return json.loads(REF.read_text())


@pytest.fixture(scope="module")
def records(ref) -> Dict:
    return ref["records"]


def _rec(records: Dict, key: str) -> Dict:
    if key not in records:
        pytest.skip(f"no Tier-1 record for {key}")
    return records[key]


def scalar_reference(key: str) -> Dict:
    """The Kuiva front-end's scalar X2C SCF for a ``systems.py`` key, as a checkpointable
    summary (``tests/stages.py``).

    ⚠ ``screening="none"``: this is a scalar quantity and the X2CAMF correction cannot move it
, so paying a four-component solve per element here would be pure cost. The key
    records it anyway, because a checkpoint that does not say what Hamiltonian produced it is
    not interpretable.
    """
    import stages
    from kuiva.interface.api import Molecule, scalar_x2c_reference

    import sys
    sys.path.insert(0, str(REPO / "tests/generate"))
    import systems as sysdef

    system = sysdef.get(key)

    def build():
        mol = Molecule(atoms=list(system.atoms), basis=system.basis,
                       charge=system.charge, spin=system.spin)
        data = scalar_x2c_reference(mol, fitting="conventional", screening="none")
        return {"converged": bool(data.converged), "e_scf": float(data.e_scf),
                "nao": int(data.nao), "nmo": int(data.nmo),
                "nelec_total": int(data.nelec_total)}

    return stages.checkpoint("scalar_scf",
                             {"system": key, "basis": system.basis, "charge": system.charge,
                              "spin": system.spin, "fitting": "conventional",
                              "screening": "none"},
                             build, extra_sources=("tests/generate/systems.py",))


# --- 1. reference integrity --------------------------------------------------------------
def test_reference_file_is_complete(records):
    """Every record converged and carries the fields the rest of the suite relies on."""
    assert records, "no Tier-1 records at all"
    for key, rec in records.items():
        assert "error" not in rec, f"{key}: {rec.get('error')}"
        assert rec["scf_converged"], f"{key}: SCF not converged"
        assert rec.get("casscf_converged", True), f"{key}: CASSCF not converged"
        assert "e_scf" in rec and "e_casscf_avg" in rec


def test_every_system_has_a_record(records):
    """No system may silently disappear from the reference file.

    The rest of the suite reaches records through ``_rec``, which *skips* when a key is
    absent. A regeneration that drops a system therefore turns its tests green by omission
    instead of red, which is precisely the failure mode a committed-reference scheme is
    supposed to be protected against. This is not hypothetical: running the generator with
    ``--fast-only`` and without ``--merge`` rewrites the file with every ``slow`` system
    (the Ce chlorides) silently removed.

    Only the *presence* of the key set is checked here; the values are the business of the
    tests below.
    """
    import sys
    sys.path.insert(0, str(REPO / "tests/generate"))
    import systems as sysdef

    # Mirrors the key construction in tests/generate/tier1_pyscf.py: one record per system
    # per basis, with the project default and the matched basis deduplicated. Systems with
    # tier1=False (a *derived* scalar counterpart — see systems.System.tier1) are exempt.
    expected = {f"{system.key}/{basis}"
                for system in sysdef.SYSTEMS if system.tier1
                for basis in dict.fromkeys([system.basis, system.basis_matched])}
    missing = sorted(expected - set(records))
    keys = ",".join(sorted({tag.split("/")[0] for tag in missing}))
    assert not missing, (
        f"{len(missing)} Tier-1 record(s) missing: {missing}\n"
        f"Regenerate them with:  python tests/generate/tier1_pyscf.py "
        f"--only {keys} --merge"
    )


def test_casscf_never_below_scf_for_closed_shell(records):
    """A CASSCF including the SCF determinant cannot lie above the SCF energy.

    Checked only for the state-specific (single-root) closed-shell cases; a state *average*
    legitimately lies above, since it averages in excited states.
    """
    for key, rec in records.items():
        if sum(rec["nroots"].values()) != 1 or rec["spin"] != 0:
            continue
        assert rec["e_casscf_avg"] <= rec["e_scf"] + E_CAS_TOL, key


# --- 2. physics the reference data must exhibit -------------------------------------------
def _degeneracy_pattern(rel_cm, tol=DEGENERACY_TOL_CM):
    blocks = []
    for e in sorted(rel_cm):
        if blocks and abs(e - blocks[-1][0]) < tol:
            blocks[-1][1] += 1
        else:
            blocks.append([e, 1])
    return [n for _, n in blocks]


@pytest.mark.parametrize("key,pattern,why", [
    ("ce3p", [7], "4f^1: the seven 2F states are degenerate without SOC"),
    ("yb3p", [7], "4f^13: one hole, likewise seven degenerate 2F states"),
    ("dy3p", [11, 7, 3], "4f^9 sextet terms 6H(11) + 6F(7) + 6P(3)"),
    ("bi", [1, 5, 3], "6p^3 terms 4S(1) + 2D(5) + 2P(3)"),
])
@pytest.mark.parametrize("basis", ["x2c-SVPall-2c", "ano-rcc-vdzp"])
def test_spin_free_degeneracy_pattern(records, key, pattern, why, basis):
    """Spin-free term degeneracies are fixed by spherical symmetry - they cannot be a matter
    of basis, code or tolerance, so they are the sharpest available check that the stored
    state-averaged CASSCF really describes the intended manifold."""
    rec = _rec(records, f"{key}/{basis}")
    assert _degeneracy_pattern(rec["e_states_rel_cm"]) == pattern, why


@pytest.mark.parametrize("basis", ["x2c-SVPall-2c", "ano-rcc-vdzp"])
def test_full_active_space_casci_equals_scf(records, basis):
    """Zn(2+) 3d^10: a CAS containing only doubly-occupied orbitals has exactly one
    determinant, so CASSCF must return the SCF energy identically. Any orbital-optimiser or
    integral-transformation bug that shifts energies shows up here immediately."""
    rec = _rec(records, f"zn2p/{basis}")
    assert rec["e_casscf_avg"] == pytest.approx(rec["e_scf"], abs=E_CAS_TOL)


@pytest.mark.parametrize("key", ["ce3p", "dy3p"])
def test_nevpt2_is_degenerate_within_a_multiplet(records, key):
    """SC-NEVPT2 corrections must be equal for states of one degenerate manifold.

    A state-dependent error in the Dyall-Hamiltonian splitting or the 4-RDM contraction
    (both flagged as error-prone) would break this while leaving total energies looking
    entirely reasonable.
    """
    rec = _rec(records, f"{key}/ano-rcc-vdzp")
    nev = rec.get("nevpt2")
    if not nev:
        pytest.skip("no NEVPT2 data in this record")
    for mult, rows in nev.items():
        corr = [r["e_corr"] for r in rows]
        assert max(corr) - min(corr) < E_NEVPT2_DEGEN_TOL, \
            f"{key} mult={mult}: NEVPT2 spread {max(corr) - min(corr):.2e} Eh over a " \
            f"degenerate manifold"


def test_yb_and_ce_f_shell_are_particle_hole_partners(records):
    """4f^1 and 4f^13 must both give a 7-fold spin-free manifold in the same active space.

    Not a deep physical identity, but it is exactly the symmetry a CI string-addressing bug
    tends to break, and Ce/Yb are in the suite as a matched pair for that reason.
    """
    ce = _rec(records, "ce3p/ano-rcc-vdzp")
    yb = _rec(records, "yb3p/ano-rcc-vdzp")
    assert ce["ncas"] == yb["ncas"] == 7
    assert ce["nelecas"] + yb["nelecas"] == 2 * 7, "not a particle-hole pair"
    for rec in (ce, yb):
        assert _degeneracy_pattern(rec["e_states_rel_cm"]) == [7]


# --- 3. Kuiva reproduces the reference ----------------------------------------------------
@pytest.mark.parametrize("key", [
    "ne", "zn2p", "hi",
    # A heavy-atom SCF is ~10 s each, over the suite's "few seconds" guard.
    pytest.param("bi", marks=pytest.mark.slow),
    pytest.param("ce3p", marks=pytest.mark.slow),
])
@pytest.mark.stage_under_test("scalar_scf")
def test_kuiva_bridge_reproduces_scf(records, key):
    """The Kuiva front-end must return the stored scalar X2C SCF energy.

    This is the one Tier-1 test that runs Kuiva code today. It is a strict regression lock:
    the bridge makes the same PySCF call, so any difference is a change in *our* ingestion.

    ⚠ **Marked as the test of ``scalar_scf``, so it always runs the SCF** . A regression lock served from a checkpoint locks the checkpoint, not the code. The
    stage is registered and the result is still *written* under ``--checkpoints=on``, so the
    CASSCF and DMRG tests to come can start from this orbital set without paying for it.
    """
    pyscf = pytest.importorskip("pyscf")

    rec = _rec(records, f"{key}/x2c-SVPall-2c")
    data = scalar_reference(key)
    assert data["converged"]
    assert data["e_scf"] == pytest.approx(rec["e_scf"], rel=E_SCF_RTOL)
    assert data["nao"] == rec["nao"]
    assert data["nelec_total"] == rec["nelectron"]


def _kuiva_scalar_casscf(key: str) -> Dict:
    """Kuiva's spinor CASSCF with **spin-orbit coupling switched off**, for a systems.py key.

    ⚠ ``with_soc=False`` is the whole point: a Kramers-paired spinor basis built from the real
    scalar MOs, with no spin-orbit term in the Hamiltonian, describes the *identical* problem
    in a doubled basis. So this is an equality, not an approximate agreement.

    ⚠ **State averaging has to match, and it is not automatic.** ``tier1_pyscf`` averages every
    spin-free root with weight ``1/n_roots``; Kuiva averages every *spinor* root uniformly,
    which weights each spin-free root by its multiplicity. The two coincide only when the
    ensemble has a single multiplicity — asserted rather than assumed below.

    ⚠ **And the count is ``scalar_states``, never ``soc_states``.** ``soc_states`` is a
    boundary of the *two-component* spectrum, whose manifolds are ``2J+1`` multiplets; with SOC
    off the manifolds are spin multiplets, and a count measured on one is generally not a
    boundary of the other. Using ``soc_states`` here put ``dy3p``'s average inside a 4-fold
    quartet block, which broke Kramers degeneracy during the optimization until the state-averaging gate
    refused — the mirror image of the C1 defect, and invisible to the odd-block gate because half
    of four is even. ``systems.System.scalar_states`` is the spin-free boundary, and where the
    two ensembles genuinely differ the record carries its own ``scalar_crosscheck`` block.
    """
    import sys

    sys.path.insert(0, str(REPO / "tests/generate"))
    import systems as sysdef                                            # noqa: PLC0415
    import tier2_kuiva                                                  # noqa: PLC0415
    from kuiva.interface.api import Molecule, casscf, spinor_reference

    system = sysdef.get(key)
    assert len(system.scalar_ensemble) == 1, \
        "{}: uniform spinor averaging equals uniform spin-free averaging only for a single " \
        "multiplicity; this system's scalar ensemble is {}".format(key, system.scalar_ensemble)

    molecule = Molecule(atoms=list(system.atoms), basis=system.basis,
                        charge=system.charge, spin=system.spin)
    # ⚠ **The Cholesky threshold is pinned here rather than inherited from the default.**
    # PySCF's CASSCF uses exact four-index integrals; Kuiva Cholesky-decomposes them, and
    # That threshold is an error bound the *user* sets. At 1e-6 the two disagree by
    # 5.8e-7 Eh on Ne — right at `E_CAS_TOL`, and entirely attributable to the factorization
    # rather than to the method this test is about. 1e-8 puts the factorization error an order
    # of magnitude below the tolerance while keeping `naux` (and every transform built on it)
    # from growing the way a 1e-10 threshold makes it grow on an f-shell system. It happens to
    # equal the current default; pinning it means this test does not move if that is
    # re-decided, which it has been once.
    reference = spinor_reference(molecule, fitting="conventional", with_soc=False,
                                 screening="none", cholesky_tol=1e-8)
    selection = tier2_kuiva.active_space_kwargs(system, reference)
    outcome = casscf(reference, n_states=system.scalar_states, max_iter=120, conv_grad=1e-5,
                     report=False, **selection)
    boundary = outcome.boundary_initial
    return {"converged": bool(outcome.converged), "energy": float(outcome.energy),
            "n_states": int(outcome.ci.energies.size),
            # ⚠ The boundary at the orbitals the optimization *started* from, because that is
            # the one that decides whether the trajectory was safe.
            "boundary_gap_cm": None if boundary is None else boundary.gap_cm,
            "boundary_clean": None if boundary is None else bool(boundary.is_clean)}


@pytest.mark.parametrize("key", ["ne", "zn2p"])
def test_kuiva_scalar_casscf_reproduces_pyscf(records, key):
    """⚠ **The tightest cross-implementation test in the suite** (done-means #2 of the CASSCF
    work): Kuiva's two-component CASSCF with SOC off against PySCF's scalar CASSCF, to 1e-7 Eh.

    With no spin-orbit term the two-component problem *is* the scalar one in a doubled basis,
    so anything but agreement at the optimizer's own convergence is a defect — there is no
    method difference left to blame it on. It is the reason Tier 1 exists alongside Tier 2: it
    separates "is the scalar CASSCF right?" from "is the SOC treatment right?".

    Measured: **8.7e-10 Eh** on ``ne`` and **1.3e-8 Eh** on ``zn2p``. ⚠ The larger one is not a
    CI difference — ``zn2p`` is a *full* active space (10 electrons in 10 spinors, one
    determinant), so its CASSCF energy **is** the SCF energy and the residual is the two SCF
    optimizations converging to slightly different points. That is why the band stays at
    ``E_CAS_TOL`` rather than being tightened to the 1e-8 the ``ne`` number would allow.
    """
    _assert_scalar_casscf_matches(records, key)


@pytest.mark.slow
@pytest.mark.parametrize("key", ["ce3p", "dy3p"])
def test_kuiva_scalar_casscf_reproduces_pyscf_f_shell(records, key):
    """The same equality on the f-shell systems, which are the expensive ones.

    ``ce3p`` is 14 state-averaged roots over 14 determinants — a full space, hence complete by
    construction. ``dy3p`` is **66** over 2002: its ⁶H term, which is the largest ensemble the
    two codes can be made to average identically, since PySCF's average is spin adapted while
    Kuiva's takes the lowest spinor roots and quartet terms fall below the ⁶P
    (``systems.System.scalar_states``). Both are far above the few-seconds bar of the default suite's timing
    guard, hence ``slow``.
    """
    _assert_scalar_casscf_matches(records, key)


def _assert_scalar_casscf_matches(records: Dict, key: str) -> None:
    """⚠ The record is looked up **in the project basis explicitly**, because Kuiva runs in
    ``System.basis``; taking whichever record came first would compare two different bases and
    fail by millihartrees for a reason that has nothing to do with the method."""
    import sys

    sys.path.insert(0, str(REPO / "tests/generate"))
    import systems as sysdef                                            # noqa: PLC0415

    system = sysdef.get(key)
    rec = _rec(records, "{}/{}".format(key, system.basis))
    # ⚠ Where the scalar ensemble differs from the SOC one the record carries its own CASSCF,
    # and taking the main ``e_casscf_avg`` would compare two different state averages — an
    # error of millihartrees dressed up as a method disagreement.
    rec = rec.get("scalar_crosscheck", rec) if system.scalar_nroots is not None else rec
    if "e_casscf_avg" not in rec:
        pytest.skip("no scalar CASSCF energy stored for {}".format(key))
    got = _kuiva_scalar_casscf(key)
    assert got["converged"], "{}: Kuiva's CASSCF did not converge".format(key)
    assert got["boundary_clean"] is not False, (
        "{}: the {}-state average starts {:.2f} cm^-1 below the first root it leaves out, so "
        "it is cutting a degenerate manifold and the comparison below is meaningless whatever "
        "it says ".format(key, system.scalar_states,
                                         got["boundary_gap_cm"] or 0.0))
    assert got["energy"] == pytest.approx(rec["e_casscf_avg"], rel=E_CAS_RTOL)


@pytest.mark.parametrize("key", ["ne", "zn2p", "hi"])
def test_kuiva_nevpt2_reproduces_reference(records, key):
    """Kuiva's SC-NEVPT2 against PySCF's strongly contracted implementation.

    Both are strongly contracted NEVPT2 on the same CASSCF reference, so agreement should be
    ~1e-6 Eh once the frozen-core treatment matches (PySCF correlates all non-active
    orbitals here).
    """
    pytest.importorskip("kuiva.pt.nevpt2",
                        reason="kuiva.pt.nevpt2 not implemented yet ")
    pytest.skip("enable once kuiva.pt.nevpt2 exposes an entry point")
