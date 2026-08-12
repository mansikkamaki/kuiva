"""Kuiva's X2C + atomic mean field against DIRAC four-component atoms.

What this tier is for
---------------------
The Tier-2 cross-code band is 15% and the error X2CAMF corrects is 16% :
that suite **cannot tell a working correction from a broken one**. These tests are the tighter
external reference, and they answer a question no in-house check can:

  A four-component Dirac-Coulomb calculation performs **no picture change at all**. So
  reproducing its one-particle spectrum from a two-component Hamiltonian *is* the statement
  that the picture-change treatment is right — and here it is stated by a **different
  program**.

⚠ The four-component number is *not* what DIRAC contributes. One is already available
in-process for free (``kuiva.amf.atomic.atomic_solution`` — and it is what
``tests/test_amf_correction.py`` compares against). What DIRAC contributes is independence,
so these tests are written to isolate exactly that: the reference was generated with the
nuclear model, speed of light, contraction and ``(SS|SS)`` treatment all pinned to what Kuiva
does, and :func:`test_the_reference_is_a_controlled_comparison` asserts that the stored
records were produced that way. Every one of those had to be pinned because DIRAC's default
differs from PySCF's — see ``tests/generate/x2camf_dirac.py``.

⚠ Which construction produced a splitting 
--------------------------------------------------------------
All splittings here come from a **self-consistent two-component SCF**
(``test_amf_correction.self_consistent_spectrum``, imported rather than copied). The
frozen-scalar-orbital construction of ``tests/test_soc_ingestion.py`` gives a *different*
number for the same operator — 908 against 1174 cm^-1 for neon — and only the self-consistent
one is comparable with a self-consistent four-component reference.

Measured agreement (dyall.v2z, uncontracted, point nucleus)
------------------------------------------------------------
========= ============== ================== =================
atom      DIRAC 4c DC    Kuiva X2C+AMF      Kuiva 1e-X2C
========= ============== ================== =================
Ne 2p       983.99 cm^-1  984.03 (+0.005%)  1262.89 (+28.3%)
Ar 3p      1628.25        1628.30 (+0.003%) 1882.44 (+15.6%)
Kr 4p      5840.96        5841.17 (+0.004%) 6306.87  (+8.0%)
Xe 5p     11432.62       11433.20 (+0.005%) 12027.54  (+5.2%)
========= ============== ================== =================

so the correction lands **three to four orders of magnitude** inside the residual it removes.
The physically meaningful tolerance for a splitting of this kind is a fraction of a percent
(a Tier-2 SOC comparison is quoted at 15%); the band asserted below is
0.5%, roughly 20x the observed spread, and is deliberately *not* tuned onto the measurement.

Cost, and what a stage checkpoint may replay here 
----------------------------------------------------------------------
Bounded by Kuiva's own four-component solve, not by the stored reference — Ne and Ar are
seconds, Kr is ~20 s (Coulomb) / ~105 s (Gaunt) and Xe several minutes, so **Kr and Xe are
marked slow** . The open-shell ions are 15-60 minutes *each*. The atomic
solutions are cached process-wide by ``kuiva.amf.atomic``, so within one session an atom is
solved once however many tests use it.

This file is where disk checkpoints earn their keep, and the split is deliberate:

* ⚠ **The four tests whose subject is the four-component solve always run it.**
  :func:`test_the_four_component_backend_matches_dirac` and the three open-shell functional
  tests are marked ``stage_under_test("amf_atomic")``. They are the like-for-like check on the
  *reference side* of every comparison here, so replaying a recorded solve into them would
  delete the check while leaving it green.
* **Everything else declares ``amf_correction`` or ``two_component_scf`` as its subject**, and
  ``amf_atomic`` is therefore upstream input that may be replayed. That is the saving: the
  15-60 minute AOC solve for Ce(3+)/Dy(3+)/Yb(3+)/Bi is skipped while the decoupling and the
  two-component SCF — the things those tests are about — are recomputed in full.

So ``pytest -m '' tests/test_x2camf_dirac.py -k "not backend_matches_dirac and not
open_shell_energy and not orbital_energies and not closed_shells_are_unaffected"`` is the
targeted re-run that costs minutes instead of hours, and a plain run of the file still pays
for the solve exactly once.
"""
import json
from pathlib import Path

import numpy as np
import pytest
from pyscf import gto

import stages
from kuiva.amf import amf_correction
from kuiva.amf.atomic import atomic_solution
from kuiva.amf.configuration import AtomicConfiguration
from test_amf_correction import self_consistent_spectrum
from test_amf_open_shell import average_of_configuration_ghf

#: Test-side sources the checkpointed builders below depend on beyond this module. Their
#: digests join the fingerprint, so changing ``self_consistent_spectrum`` invalidates every
#: two-component spectrum recorded with it (``tests/stages.py``).
HELPER_SOURCES = ("tests/test_amf_correction.py", "tests/test_amf_open_shell.py")

REPO = Path(__file__).resolve().parents[1]
REFERENCE = REPO / "tests/reference/x2camf_dirac.json"
BASIS_CROSSCHECK = REPO / "tests/reference/basis_crosscheck.json"

HARTREE_CM = 219474.6313632
#: PySCF's name for DIRAC's ``dyall.v2z``. Content-matched, not name-matched — see
#: :func:`test_the_basis_is_content_matched_not_name_matched`.
BASIS = "dyallv2z"
INTERACTION = {"dc": "coulomb", "dcg": "gaunt"}

# --- tolerances, each with the physically meaningful figure beside it ----------------------

#: Four-component total energy, Kuiva's PySCF backend against DIRAC. Meaningful: 1e-8 Eh
#: . Observed 1.3e-9 (Ne), 1.6e-8 (Ar), 3.3e-7 (Kr), 2.0e-6 (Xe),
#: 1.6e-5 (Bi) — it grows with the basis because the two codes project their near-singular
#: four-component metric at different thresholds (``pyscf_dhf.METRIC_LINDEP_THRESHOLD``),
#: which is a numerical, not a physical, difference.
#:
#: ⚠ **The bound is therefore RELATIVE, and that is a user decision recorded as a user decision.** It
#: was absolute (1e-5 Eh) and Bi failed it at 1.59e-5 — while every other atom in the set,
#: including the open-shell Ce(3+)/Dy(3+)/Yb(3+), passed. The error is relative and the bound
#: was not: at 2.2e+4 Eh Bi is three times the total energy of Xe, so an absolute bound
#: calibrated on Ne-through-Xe cannot cover it, and the next heavier atom would have failed
#: too. Observed relative: 1.0e-11 (Ne), 3.0e-11 (Ar), 1.2e-10 (Kr), 2.7e-10 (Xe),
#: 7.4e-10 (Bi), so the bound below is ~14x the worst case and is not tuned onto it.
#:
#: ⚠ **This bound is NOT a convergence claim, and it cannot be read as one** (user decision):
#: the two codes are not evaluating exactly the same quantity, because they project the metric
#: differently. At Bi, 1e-8 relative *is* 2.2e-4 Eh absolute — four orders above the 1e-8 Eh
#: counts as physically meaningful. That is the honest price of a cross-code check against a
#: slightly different definition, and the tight statements about this method live elsewhere:
#: the j-splitting bands below, and the energy functional of ``test_amf_decouple.py`` (3.4e-07
#: Eh for Ne), which compares Kuiva against Kuiva and therefore *can* be held to it.
#:
#: ⚠ Do not read a failure here as a failure of the *correction*: when Bi failed the absolute
#: bound, its j-splitting, its whole occupied spectrum and its interaction tracking all passed.
#: This assertion is the like-for-like check on the **reference side** of the comparison.
FOUR_COMPONENT_ENERGY_RTOL = 1e-8

#: Four-component occupied spinor energies, same comparison. Observed max 1.1e-6 Eh (Ne),
#: falling to 4.3e-7 at Xe — the projection difference shows up in the total, not in the
#: individual orbital energies.
FOUR_COMPONENT_SPINOR_TOL_EH = 1e-5

#: X2C+AMF valence j-splitting against the four-component one, **closed shell**. Observed
#: 0.003-0.031%, i.e. the band below is ~20x the worst case and is deliberately not tuned
#: onto it.
SPLITTING_TOL_RELATIVE = 5e-3

#: ⚠ The same quantity for an **open** shell, and it is four times looser for a reason worth
#: stating rather than hiding in a constant. Measured, all against DIRAC in ``dyall.v2z`` with
#: the average-of-configuration functional now correct on both sides:
#:
#: ===== ========= ========= ========== =========
#: atom  DIRAC     X2C+AMF   rel error  abs error
#: ===== ========= ========= ========== =========
#: Ne     983.99    984.03     0.005%    0.045
#: Ar    1628.25   1628.30     0.003%    0.045
#: C       54.34     53.96     0.690%    0.375
#: O      296.44    295.79     0.220%    0.654
#: Ti3+   345.48    343.67     0.523%    1.809
#: ===== ========= ========= ========== =========
#:
#: The open-shell residual is 40x the closed-shell one in *absolute* terms too, so this is not
#: a small splitting flattering a relative measure — it is a genuine property of the
#: approximation. **An atomic mean field over a partially filled valence shell represents the
#: two-electron picture change less well than one over a closed shell**, which is physically
#: unsurprising (the picture change is a core effect and the open shell is the part the mean
#: field describes worst) and had no way of being seen before this reference set existed.
#:
#: ⚠ **Open question, deliberately not chased here.** A correction carrying the
#: average-of-configuration coupling in its own mean fields might close it — but there is no
#: single mean field to picture-change once the shells have different Fock operators
#: (``E_2 = 1/2 Tr[D G[D]] + (alpha-1)/2 Tr[D_o G[D_o]]`` is not ``1/2 Tr[D G_eff]`` for any
#: one ``G_eff``), so plan option (b) is not even well defined without a further choice. That
#: is an argument for option (a), which is what is implemented; it is not a proof that the
#: residual is irreducible.
OPEN_SHELL_SPLITTING_TOL_RELATIVE = 2e-2

#: How much better the corrected spectrum must be than the uncorrected one. Observed 2760x
#: (Ne) falling to 309x (Xe) — the picture change is a core effect and the core spinors grow
#: faster than the correction's residual does. This is the assertion that matters most: it is
#: a statement about the *correction*, and unlike an absolute band it cannot be satisfied by a
#: basis coincidence.
SPECTRUM_IMPROVEMENT_FACTOR = 100.0

#: The one-electron error the correction exists to remove, as a fraction of the splitting.
#: Asserted as a floor so that a test which passed because *both* Hamiltonians were wrong
#: could not look like success. ⚠ Observed **+5.2% at Xe/DC** and +40% at Ne/DCG — the floor
#: is 3% rather than 5% because the smallest case would otherwise sit 4% away from failing,
#: and the trend is real: the *relative* one-electron error falls with Z (the nuclear
#: spin-orbit term grows faster than the two-electron screening of it). A heavier atom would
#: need this lowered again, not the test deleted.
ONE_ELECTRON_ERROR_FLOOR = 0.03

CASES = [
    ("Ne", "dc"), ("Ne", "dcg"),
    ("Ar", "dc"), ("Ar", "dcg"),
    pytest.param("Kr", "dc", marks=pytest.mark.slow),
    pytest.param("Kr", "dcg", marks=pytest.mark.slow),
    pytest.param("Xe", "dc", marks=pytest.mark.slow),
    pytest.param("Xe", "dcg", marks=pytest.mark.slow),
]

#: Open-shell cases. ⚠ These were `xfail(strict=True)` when the reference set was first
#: generated, and the marks are gone because the defect they recorded is fixed (now fixed): Kuiva's four-component "average of configuration" was really
#: fractional-occupation Hartree-Fock, with an open-open two-electron coupling of ``(q/n)^2``
#: where a true configuration average has ``q(q-1)/(n(n-1))``. **`strict=True` is what forced
#: the issue** — the fix turned them into XPASS failures, so the marks could not outlive the
#: defect.
#:
#: C and O are seconds and stay in the default suite; the ions need an uncontracted
#: four-component AOC solve of minutes to tens of minutes each and are `slow`.
OPEN_SHELL_CASES = [
    ("C", "dc"),
    ("O", "dc"),
    pytest.param("Ti", "dc", marks=pytest.mark.slow),
    pytest.param("Ce", "dc", marks=pytest.mark.slow),
    pytest.param("Dy", "dc", marks=pytest.mark.slow),
    pytest.param("Yb", "dc", marks=pytest.mark.slow),
    pytest.param("Bi", "dc", marks=pytest.mark.slow),
]
CASES = CASES + OPEN_SHELL_CASES


def reference():
    if not REFERENCE.is_file():
        pytest.skip("run tests/generate/x2camf_dirac.py to create {}".format(REFERENCE.name))
    return json.loads(REFERENCE.read_text())


def record(symbol, hamiltonian):
    data = reference()
    key = "{}/{}".format(symbol, hamiltonian)
    rec = data["records"].get(key)
    if rec is None or rec.get("status") != "ok":
        pytest.skip("no usable DIRAC record for {}".format(key))
    return rec


def stage_key(rec, hamiltonian=None, **extra):
    """The checkpoint key for a record: everything that determines the calculation.

    Taken from the **stored record**, for the same reason :func:`species` is: the charge and
    the reference configuration are two spellings of one fact, and a key that kept its own
    copy could label a Ce(3+) result with a neutral-Ce key.
    """
    configuration = AtomicConfiguration.parse(rec["configuration"]) \
        if rec.get("configuration") else AtomicConfiguration.ground(rec["element"])
    key = {"element": rec["element"], "charge": rec.get("charge", 0),
           "configuration": configuration.canonical, "basis": BASIS, "uncontract": True}
    if hamiltonian is not None:
        key["interaction"] = INTERACTION[hamiltonian]
    key.update(extra)
    return key


def four_component(rec, hamiltonian):
    """Kuiva's own four-component atomic solution, reduced to what this file asserts on.

    ⚠ The payload is the *summary*, not the solution: the four block groups of a decontracted
    lanthanide are ~1 GB and nothing here reads them (``kuiva/amf/cache.py`` declines to store
    them for the same reason). Anything that needs the blocks must call
    :func:`kuiva.amf.atomic.atomic_solution` directly and pay for it.
    """
    def build():
        mol, configuration = species(rec)
        solution = atomic_solution(rec["element"], mol._basis[rec["element"]],
                                   configuration=configuration,
                                   interaction=INTERACTION[hamiltonian])
        return {"converged": bool(solution.converged), "e_tot": float(solution.e_tot),
                "charge": int(solution.charge),
                "spinor_energies": np.asarray(solution.occupied_energies())}

    return stages.checkpoint("amf_atomic", stage_key(rec, hamiltonian), build)


def species(rec):
    """``(mol, configuration)`` for a record, built from what the record says it is.

    ⚠ The charge and the configuration come from the **stored record**, not from a second
    table here. They are two spellings of one fact — the rule: "the configuration is
    the single source of truth for the charge state" — and a test that kept its own copy would
    be free to compare Kuiva's Ce(3+) against DIRAC's neutral Ce.
    """
    symbol, charge = rec["element"], rec.get("charge", 0)
    configuration = AtomicConfiguration.parse(rec["configuration"]) \
        if rec.get("configuration") else AtomicConfiguration.ground(symbol)
    assert configuration.n_electrons == int(gto.charge(symbol)) - charge, rec["element"]
    mol = gto.M(atom=[(symbol, (0.0, 0.0, 0.0))], basis=BASIS, charge=charge,
                spin=configuration.n_electrons % 2, verbose=0)
    return mol, configuration


def _spectrum(mol, configuration, correction, open_shell):
    """Occupied two-component spinor energies, by the construction the species needs.

    ⚠ **An open shell needs the average-of-configuration SCF and the closed-shell one does not
    merely give a worse answer — it does not converge.** A plain aufbau GHF on an open-shell
    ion picks arbitrarily among a degenerate frontier manifold; measured on Ti(3+), the
    assertion inside ``self_consistent_spectrum`` fires rather than a wrong number coming
    back. Both sides of this comparison must occupy the same averaged state, and the filling
    rule is shared with the four-component backend so they provably do
    (``kuiva.amf.configuration.average_occupations``).
    """
    if not open_shell:
        return self_consistent_spectrum(mol, correction)
    from pyscf.x2c import x2c

    from kuiva.spinor.expand import decompose_two_component, two_component_operator

    helper = x2c.SpinOrbitalX2CHelper(mol)
    h = two_component_operator(*decompose_two_component(np.asarray(helper.get_hcore())))
    if correction is not None:
        h = h + correction.hamiltonian()
    mf = average_of_configuration_ghf(mol, configuration, h)
    assert mf.converged, "the two-component AOC SCF did not converge"
    occ = np.asarray(mf.mo_occ)
    return np.sort(np.asarray(mf.mo_energy)[occ > 1e-12])


def spectra(symbol, hamiltonian):
    """``(uncorrected, corrected)`` occupied two-component spinor energies [Eh].

    Both halves go through a ``two_component_scf`` checkpoint, and the *uncorrected* one is
    keyed without an interaction because it is the plain one-electron X2C Hamiltonian and does
    not know which two-electron operator the correction was going to be built with. The
    per-session reuse the two module-level dictionaries here used to provide is now the memo
    inside ``stages.checkpoint``, so there is one mechanism instead of two.
    """
    rec = record(symbol, hamiltonian)
    open_shell = bool(rec.get("open_shell"))

    def uncorrected():
        mol, configuration = species(rec)
        return {"spinors": _spectrum(mol, configuration, None, open_shell)}

    def corrected():
        mol, configuration = species(rec)
        # ⚠ The correction's reference configuration is the ion's own, matching what DIRAC
        # was run in. That is *not* Kuiva's default for the f block (M(3+) is, and here they
        # coincide) nor for Ti (whose default is the neutral atom), so it is passed explicitly
        # rather than left to the default — the measured sensitivity is only 0.21%, but a
        # cross-code comparison should not be spending it.
        correction = amf_correction(mol, method="x2camf", configuration=configuration,
                                    interaction=INTERACTION[hamiltonian])
        return {"spinors": _spectrum(mol, configuration, correction, open_shell)}

    plain = stages.checkpoint("two_component_scf", stage_key(rec, correction="none"),
                              uncorrected, extra_sources=HELPER_SOURCES)
    screened = stages.checkpoint("two_component_scf",
                                 stage_key(rec, hamiltonian, correction="x2camf"),
                                 corrected, extra_sources=HELPER_SOURCES)
    return plain["spinors"], screened["spinors"]


def splitting_cm(energies, n_valence):
    valence = energies[-n_valence:]
    return float(valence[-1] - valence[0]) * HARTREE_CM


# --- the reference itself: what makes a cross-code number worth anything -------------------

def test_the_reference_is_a_controlled_comparison():
    """The stored records must have been produced with every variable except *which program*
    pinned to what Kuiva does.

    This is not bookkeeping. DIRAC 26.1 defaults to a **Gaussian** nuclear charge distribution
    and CODATA-2022 constants, PySCF to a point nucleus and CODATA 2018; a reference
    regenerated without the pinning keywords would differ from Kuiva by a real physical effect
    that grows with ``Z`` and would be read as picture-change error. So the check is on the
    *parsed DIRAC output* — what DIRAC said it did — not on the input we believe we wrote.
    """
    data = reference()
    assert data["controlled"]["speed_of_light"] == pytest.approx(137.035999084, abs=1e-9)
    for key, rec in data["records"].items():
        if rec.get("status") != "ok":
            continue
        assert rec["nuclear_model"] == "point charge", key
        assert rec["speed_of_light"] == pytest.approx(137.035999084, abs=1e-8), key
        assert rec["uncontracted"] is True, key
        assert rec["basis"] == "dyall.v2z", key


def test_the_basis_is_content_matched_not_name_matched():
    """``dyall.v2z`` (DIRAC) and ``dyallv2z`` (PySCF) must be the *same functions*.

    Two programs agreeing on a basis-set *name* is not evidence of anything (the ``"6p"`` AO-label incident is the same class of silent,
    basis-dependent error). The primitive exponents are compared in
    ``tests/generate/crosscheck_external_basis.py``; this asserts that the comparison covers
    the atoms this reference actually uses.
    """
    if not BASIS_CROSSCHECK.is_file():
        pytest.skip("run tests/generate/crosscheck_external_basis.py")
    crosscheck = json.loads(BASIS_CROSSCHECK.read_text())["dyall"]
    for symbol in sorted({rec["element"] for rec in reference()["records"].values()}):
        case = crosscheck.get("dyallv2z/{}".format(symbol))
        assert case is not None, (
            "{} is in the DIRAC reference but its dyallv2z primitives were never checked "
            "against DIRAC's dyall.v2z".format(symbol))
        assert case["ok"] is True, symbol


def test_records_carry_a_complete_occupied_spectrum():
    """The occupied spinors must account for every electron the configuration has.

    The generator already refuses a record that does not, because DIRAC fills the lowest
    spinors of each fermion irrep from the ``.CLOSED SHELL`` / ``.OPEN SHELL`` counts
    **without checking them against the electron configuration** — a wrong split converges
    silently to the wrong state (the trap ``tests/generate/tier2_dirac.py`` records). Asserted
    again here so that the committed file, not just the generator run, is known to be sound.

    ⚠ For an open shell the **electron** count and the **spinor** count are different numbers
    (Ti(3+): 19 electrons over 28 partly occupied spinors), and both are checked. Only the
    first fixes the charge; only the second says the average covered the whole ``nl`` shell
    rather than a ``j`` sub-shell, which is a distinction no symmetry check can make because
    both are spherical .
    """
    for key, rec in reference()["records"].items():
        if rec.get("status") != "ok":
            continue
        electrons = rec["atomic_number"] - rec.get("charge", 0)
        spinors = electrons + rec.get("open_spinors", 0) - rec.get("open_electrons", 0)
        # ⚠ The occupation is read from DIRAC's printout (``f = 0.3333``, four decimals), so
        # a 2p2 shell sums to 5.9998. The tolerance is that rounding, not a physical one.
        assert rec.get("n_electrons", electrons) == pytest.approx(
            electrons, abs=1e-4 * max(rec["n_occupied_spinors"], 1)), key
        assert rec["n_occupied_spinors"] == spinors, key
        assert len(rec["spinor_energies"]) == spinors, key
        assert rec["spinor_energies"] == sorted(rec["spinor_energies"]), key


# --- Kuiva's four-component backend against a different program ---------------------------

@pytest.mark.stage_under_test("amf_atomic")
@pytest.mark.parametrize("symbol,hamiltonian", CASES)
def test_the_four_component_backend_matches_dirac(symbol, hamiltonian):
    """``kuiva.amf``'s PySCF Dirac-Hartree-Fock against DIRAC's, same everything.

    This is a like-for-like on the *reference* side of the comparison, and it has to pass
    before the X2C+AMF comparison below means anything: if Kuiva's four-component solve
    disagreed with DIRAC's, an agreement of the two-component result with either would be
    uninterpretable. It is also the strongest available check on the Stage-1 backend — the
    metric projection and the by-energy occupation selection it needs 
    are both silent failure modes, and a 200 Eh error was once reported as nothing but
    ``converged = False``.

    ⚠ **Marked as the test of ``amf_atomic``, so it is never served a checkpoint**.
    This is the whole four-component reference side of the file; replaying a recorded solve
    into it would compare last week's Kuiva against DIRAC and pass whatever the backend now
    does. It is also the one test here that is sensitive to the things a source fingerprint
    cannot see — the basis-set data, MKL, the metric projection — which is exactly what a
    cross-code check is for.
    """
    rec = record(symbol, hamiltonian)
    solution = four_component(rec, hamiltonian)
    assert solution["converged"]
    assert solution["e_tot"] == pytest.approx(rec["e_total"], rel=FOUR_COMPONENT_ENERGY_RTOL)
    ours = solution["spinor_energies"]
    theirs = np.asarray(rec["spinor_energies"])
    assert ours.size == theirs.size
    assert np.max(np.abs(ours - theirs)) < FOUR_COMPONENT_SPINOR_TOL_EH


# --- the actual four-component statement --------------------------------------------------

@pytest.mark.stage_under_test("two_component_scf")
@pytest.mark.parametrize("symbol,hamiltonian", CASES)
def test_x2camf_reproduces_the_four_component_j_splitting(symbol, hamiltonian):
    """The valence j-splitting from X2C + AMF against DIRAC's four-component one.

    Both bounds matter. The upper one says the correction is right; the lower one says there
    was something to correct — without it, a test in which *both* Hamiltonians were wrong in
    the same way would read as success.
    """
    rec = record(symbol, hamiltonian)
    n = rec["valence_spinors"]
    uncorrected, corrected = spectra(symbol, hamiltonian)
    four_component = rec["valence_splitting_cm"]

    corrected_error = abs(splitting_cm(corrected, n) - four_component) / four_component
    one_electron_error = abs(splitting_cm(uncorrected, n) - four_component) / four_component
    tolerance = (OPEN_SHELL_SPLITTING_TOL_RELATIVE if rec.get("open_shell")
                 else SPLITTING_TOL_RELATIVE)
    assert corrected_error < tolerance, (
        "{} {}: X2C+AMF splitting {:.2f} cm^-1 against DIRAC {:.2f}".format(
            symbol, hamiltonian, splitting_cm(corrected, n), four_component))
    assert one_electron_error > ONE_ELECTRON_ERROR_FLOOR, (
        "the one-electron operator is supposed to be badly wrong here; if it is not, this "
        "atom no longer discriminates and the case should be replaced")


@pytest.mark.parametrize("symbol,hamiltonian", CASES)
def test_x2camf_reproduces_the_whole_occupied_spectrum(symbol, hamiltonian):
    """Not just the splitting: every occupied spinor energy, core included.

    A splitting is a difference and can be right for compensating reasons. The absolute
    orbital energies cannot: the core spinors carry the picture change at its largest (the
    1e-X2C error on Kr's 1s is 0.33 Eh), so this is the sharper statement of the two, and it
    is the same quantity DIRAC's own amfX2C tutorial reports its accuracy in.
    """
    rec = record(symbol, hamiltonian)
    uncorrected, corrected = spectra(symbol, hamiltonian)
    four_component = np.asarray(rec["spinor_energies"])
    assert corrected.size == four_component.size

    corrected_mae = float(np.mean(np.abs(corrected - four_component)))
    uncorrected_mae = float(np.mean(np.abs(uncorrected - four_component)))
    assert corrected_mae * SPECTRUM_IMPROVEMENT_FACTOR < uncorrected_mae, (
        "{} {}: corrected MAE {:.2e} Eh vs uncorrected {:.2e} Eh — the correction is not "
        "buying the orders of magnitude it should".format(
            symbol, hamiltonian, corrected_mae, uncorrected_mae))


# --- the open-shell functional, which this reference set fixed ------------------------------

#: ``(symbol, q, n)`` — electrons in the open shell and spinors it spans.
_OPEN_SHELLS = (("C", 2, 6), ("O", 4, 6))


def _coupling(q, n):
    """``alpha = n(q-1)/(q(n-1))``, the open-open coupling of a true configuration average
    relative to what a fractional density gives. See
    :attr:`kuiva.amf.configuration.OpenShell.coupling`."""
    return (n * (q - 1)) / float(q * (n - 1))


@pytest.mark.stage_under_test("amf_atomic")
@pytest.mark.parametrize("symbol,q,n", _OPEN_SHELLS)
def test_the_open_shell_energy_functional_is_a_true_configuration_average(symbol, q, n):
    """⚠ **The regression guard on the fix this reference set forced** .

    Occupying a frontier shell fractionally is right; evaluating the two-electron energy over
    the resulting *density* is not, because it factorizes the pair average ``<n_i n_j>`` into
    ``<n_i><n_j>``. Kuiva did the second, and its open-shell four-component energies were
    **0.30-0.47 Eh above DIRAC's** while every closed-shell record in the same file agreed to
    1e-9 in the same basis through the same code path.

    ⚠ **No in-house check could have found it, and none did.** The in-house energy functional
    compares Kuiva's two-component AOC SCF against Kuiva's own four-component one, and both
    carried the same functional, so it agreed to 3e-09 Eh while both were wrong. A fractional
    density is perfectly spherical, so the anisotropy guard, the time-reversal residual and
    hermiticity are all silent by construction. It took a different program computing the
    same state under the correct functional.

    Asserted at the **closed-shell** tolerance, because that is what the fix bought: C and O
    now agree with DIRAC to 1e-10 and 5e-10 Eh.
    """
    rec = record(symbol, "dc")
    solution = four_component(rec, "dc")
    assert solution["converged"]
    assert 0.0 <= _coupling(q, n) < 1.0            # an open shell, and not a closed one
    assert solution["e_tot"] == pytest.approx(rec["e_total"], rel=FOUR_COMPONENT_ENERGY_RTOL)


@pytest.mark.stage_under_test("amf_atomic")
def test_the_open_shell_orbital_energies_use_the_same_convention_as_dirac():
    """⚠ **For an open shell the orbital energies are a convention, and it is worth +/-15%.**

    The diagonal blocks of the effective Fock cannot change the density or the total energy —
    every orbital within a shell has the same occupation, so a rotation among them leaves both
    invariant — but they fix what the orbital *energies* are. Measured against DIRAC over the
    whole occupied spectrum of C, O and Ti(3+), with the density identical in all three cases:

    ==========================================  ==========
    each shell from its own Fock                 MAE 1.0e-01 Eh
    everything from ``(F_c + F_o)/2``            MAE 1.0e-01 Eh
    closed from ``F_c``, open from the average   MAE 2.1e-08 Eh
    ==========================================  ==========

    Seven orders of discrimination across ``alpha`` = 0, 0.6 and 0.9, so this identifies
    Roothaan's canonical convention rather than fitting one. ⚠ It is **not** Janak's theorem,
    which gives ``dE/dn_t = F_o`` for the open shell; both are defensible and the comparison
    merely has to be consistent. For a closed shell the two coincide identically, which is why
    the question never arose before this reference set existed.

    This test is what stops the convention drifting: the *splitting* is what the reference chain quotes, and
    a change of convention would move it by 15% at carbon with nothing else looking wrong.
    """
    for symbol, _, _ in _OPEN_SHELLS:
        rec = record(symbol, "dc")
        ours = four_component(rec, "dc")["spinor_energies"]
        theirs = np.asarray(rec["spinor_energies"])
        assert ours.size == theirs.size, symbol
        assert np.max(np.abs(ours - theirs)) < FOUR_COMPONENT_SPINOR_TOL_EH, symbol


@pytest.mark.stage_under_test("amf_atomic")
def test_closed_shells_are_unaffected_by_the_open_shell_machinery():
    """The closed-shell path must be **bitwise** what it was, and it is by construction.

    With no open shell there is one Fock and the block rule of
    :func:`kuiva.amf.configuration.install_configuration_average` gives ``F_eff = F_c``
    everywhere, so nothing is installed at all and every committed reference number stands untouched.

    Asserted beside the open-shell tests because a bare "open shells now agree" is compatible
    with having broken something else in the same file; "open shells agree to 1e-10 *and*
    closed shells still agree to 1e-9 through the same code path" is not.
    """
    for symbol in ("Ne", "Ar"):
        rec = record(symbol, "dc")
        assert four_component(rec, "dc")["e_tot"] == pytest.approx(
            rec["e_total"], rel=FOUR_COMPONENT_ENERGY_RTOL), symbol


# --- the decomposition this reference exists to provide ------------------------------------

def test_gaunt_lowers_the_splitting_and_its_relative_effect_shrinks_with_z():
    """The DC-vs-DCG decomposition, asserted on the stored records.

    Two separate statements, and the second is the one people get backwards. Adding Gaunt
    *lowers* the four-component valence splitting, because spin-other-orbit is a second
    screening channel. But its **relative** size *shrinks* with ``Z`` (-8.2% Ne, -3.9% Ar,
    -1.8% Kr, -1.2% Xe), which is the opposite of the naive expectation: one-electron nuclear
    spin-orbit coupling grows far faster than two-electron screening does.

    ⚠ The consequence is worth stating where it will be read, because it decides how this
    reference should be used: the DC-vs-DCG decomposition is most informative at the **light**
    end and matters least for the heavy target systems. A 1% Xe number here is
    physics, not a failed cross-check.
    """
    data = reference()["records"]
    effects = []
    for symbol in ("Ne", "Ar", "Kr", "Xe"):
        dc, dcg = data.get("{}/dc".format(symbol)), data.get("{}/dcg".format(symbol))
        if not (dc and dcg) or dc.get("status") != "ok" or dcg.get("status") != "ok":
            continue
        relative = (dcg["valence_splitting_cm"] - dc["valence_splitting_cm"]) \
            / dc["valence_splitting_cm"]
        assert relative < 0, "{}: Gaunt must reduce the valence splitting".format(symbol)
        effects.append((dc["atomic_number"], relative))
    if len(effects) < 2:
        pytest.skip("need at least two atoms with both interactions")
    effects.sort()
    magnitudes = [abs(r) for _, r in effects]
    assert magnitudes == sorted(magnitudes, reverse=True), (
        "the relative Gaunt effect must fall monotonically with Z, got {}".format(effects))


@pytest.mark.parametrize("symbol,hamiltonian", CASES)
def test_the_correction_tracks_the_interaction_it_was_built_with(symbol, hamiltonian):
    """An AMF correction built with Gaunt must match the Gaunt reference *better* than a
    Coulomb-only one does, and vice versa.

    This is what makes the ``interaction`` argument meaningful rather than decorative: it is
    possible to build a correction that improves every splitting without ever responding to
    which two-electron operator it was asked for, and every other test here would pass.
    """
    rec = record(symbol, hamiltonian)
    other = "dcg" if hamiltonian == "dc" else "dc"
    record(symbol, other)                    # skips the test if the other record is missing
    n = rec["valence_spinors"]
    matched = splitting_cm(spectra(symbol, hamiltonian)[1], n)
    mismatched = splitting_cm(spectra(symbol, other)[1], n)
    target = rec["valence_splitting_cm"]
    assert abs(matched - target) < abs(mismatched - target), (
        "{}: the {} correction ({:.2f} cm^-1) is no closer to the {} reference ({:.2f}) than "
        "the {} one ({:.2f}) is".format(symbol, INTERACTION[hamiltonian], matched,
                                        hamiltonian.upper(), target, INTERACTION[other],
                                        mismatched))
