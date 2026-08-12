"""Kuiva's X2CAMF against the authors' own implementation.

What this tier adds to the DIRAC one
------------------------------------
``tests/test_x2camf_dirac.py`` says, from outside and to 0.003-0.005% on a j-splitting, that
Kuiva's *answer* is right. It cannot say **where** a disagreement would live, because a
four-component calculation performs no picture change and so has no term corresponding to any
of Kuiva's. The ``x2camf`` plugin — the reference implementation by the group that published
the method (Liu & Cheng, JCP 148, 144108 (2018)) — does, and that is the only thing it is
here for.

⚠ **This is not a same-quantity comparison until one thing is said**, and it was found by
measurement rather than assumed (see :mod:`kuiva.amf.x2camf_plugin`): **the plugin's default
entry point returns only the spin-dependent half.** ``x2camf.amfi`` without ``pcc=True``
carries no two-electron *scalar* picture change at all (measured for Ne: ``max |dh_sf|``
1.0e-05 against Kuiva's 3.0e-02), and on the energy functional it is **indistinguishable from
no correction**. That is not a disagreement, it is a different quantity, and
:func:`test_the_soc_only_variant_is_not_a_hamiltonian_correction` pins it so nobody adopts it
as one.

Measured agreement (neutral reference, same primitive basis, ``pcc`` variant)
------------------------------------------------------------------------------
==================== ==================== ==================== =====================
quantity             primitive basis      molecular basis      what it means
==================== ==================== ==================== =====================
``dw`` (spin-orbit)  4.2e-10 - 6.7e-09    2.9e-10 - 1.6e-08    the same matrix
``dh_sf`` (scalar)   4.4e-10 - 1.0e-08    5.0e-10 - 3.0e-08    the same matrix
j-splitting          -                    exact to 0.01 cm^-1  the same observable
==================== ==================== ==================== =====================

so the two implementations are now **numerically identical to eight or nine digits** — which
is a far stronger statement than any tolerance band, and it is worth recording *why* it is
available, because it was not always so.

⚠ **This file used to assert a 5-10% band on ``dh_sf`` in the primitive basis, and the band
was real.** The two codes differed by exactly one documented choice: which ``X`` the
two-electron picture change is decoupled with. The plugin uses the converged four-component
**Fock**'s and compensates with ``h1e(X_2e) - h1e(X_1e)``; Kuiva used the **one-electron**
``X`` at first. Kuiva adopted the plugin's convention (a recorded adoption,
:func:`kuiva.amf.decouple.x2c_decoupling`) on the evidence the band itself produced — the
Fock convention is 5-35x sharper on the energy functional and moves no splitting — so what
remained of the difference is now rounding.

⚠ **A test suite that agrees to machine precision has lost a discriminator, so the loss is
compensated for deliberately.** Agreement this tight can also be produced by two code paths
that have quietly become one, which is why
:func:`test_the_agreement_is_between_two_independent_implementations` asserts that the
plugin's numbers really did come from the plugin, and why
:func:`test_the_correction_is_a_genuine_fraction_of_the_one_electron_error` keeps a floor
under the size of the thing being agreed about.

On the energy functional — the check that discriminates the subtraction — both now land at
-1.1e-07 Eh for Ne and -1.1e-06 for Ar, four to five orders inside the +7.1e-03 / +5.0e-02 Eh
that plain one-electron X2C misses by.

Running
-------
The stored records are asserted always. The **live** comparison runs only when the plugin is
importable (``scripts/bootstrap/80_x2camf.sh``) and is skipped otherwise, so this file is
green on a machine that has never built it.
"""
import json
from pathlib import Path

import numpy as np
import pytest
from pyscf import gto
from pyscf.gto import mole

from kuiva.amf import amf_correction
from kuiva.amf.configuration import AtomicConfiguration
from kuiva.amf.correction import METHODS
from kuiva.amf import x2camf_plugin as plugin

REPO = Path(__file__).resolve().parents[1]
REFERENCE = REPO / "tests/reference/x2camf_plugin.json"

# --- tolerances, each with the physically meaningful figure beside it ----------------------

#: Either part of the correction, Kuiva against the plugin, relative to the larger of the two.
#: Observed 4.2e-10 to 3.0e-08 across both bases, both interactions and both parts, since the
#: two adopted the same decoupling convention (module docstring). Two independent codes
#: agreeing at 1e-08 are doing the same arithmetic in a different order; the bound is ~30x the
#: worst case, which leaves room for a compiler or a BLAS to reassociate and none for a
#: convention to drift.
#:
#: ⚠ This is **three orders tighter** than the 5e-03 it replaced, and deliberately so: a
#: tolerance kept at the old value would no longer be able to see the difference it was
#: written to describe coming back.
CORRECTION_TOL_RELATIVE = 1e-6

#: Valence j-splitting, Kuiva against the plugin, from the same self-consistent two-component
#: SCF. Observed exact to the printed 0.01 cm^-1. The physically meaningful figure for a
#: splitting of ~1600 cm^-1 is ~1 cm^-1, so this bound is well inside anything that could
#: matter and still far above the observation.
SPLITTING_TOL_RELATIVE = 1e-5

#: How much better the energy functional must be with either correction than with none, in the
#: same basis against the same four-component solve. Observed 104x (Kuiva, Ar/dyallv2z) to
#: 66000x (plugin, Ne/x2c-SVPall-2c).
ENERGY_IMPROVEMENT_FACTOR = 50.0


def reference():
    if not REFERENCE.is_file():
        pytest.skip("{} not generated (tests/generate/x2camf_plugin.py)".format(
            REFERENCE.name))
    return json.loads(REFERENCE.read_text())


def records():
    return reference()["records"]


def record(key):
    r = records().get(key)
    if r is None:
        pytest.skip("no stored record for {}".format(key))
    if r.get("status") != "ok":
        pytest.fail("stored record {} is not ok: {}".format(key, r.get("status")))
    return r


KEYS = ["Ne/x2c-SVPall-2c/coulomb", "Ne/x2c-SVPall-2c/gaunt",
        "Ne/dyallv2z/coulomb", "Ne/dyallv2z/gaunt",
        "Ar/x2c-SVPall-2c/coulomb", "Ar/x2c-SVPall-2c/gaunt",
        "Ar/dyallv2z/coulomb", "Ar/dyallv2z/gaunt"]

needs_plugin = pytest.mark.skipif(not plugin.available(),
                                  reason="the x2camf plugin is not installed "
                                         "(scripts/bootstrap/80_x2camf.sh)")


# --- the stored comparison ----------------------------------------------------------------

@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("part", ["dw", "dh_sf"])
def test_the_two_implementations_produce_the_same_matrix(key, part):
    """The correction term by term, from two independent implementations of one paper.

    This is the most valuable assertion in the file: the two codes share no line of source, no
    basis-handling path and no four-component solver, and they agree to 1e-08 on **both** parts
    in **both** bases.

    ``dw`` is what changes a spin-orbit splitting and ``dh_sf`` is the larger half that
    Breit-Pauli AMFI and SNSO do not describe at all, so neither is optional.
    """
    r = record(key)
    for where in ("primitive_basis", "molecular_basis"):
        rel = r[where]["kuiva_vs_plugin_pcc"]["rel_" + part]
        assert rel < CORRECTION_TOL_RELATIVE, "{} {}: rel {} = {:.2e}".format(
            key, where, part, rel)


@pytest.mark.parametrize("key", KEYS)
def test_the_splittings_agree(key):
    """The observable, which is what any of this is for.

    ⚠ This test survives from when the two codes decoupled with a different ``X`` and it
    asserted that a real 5-10% difference in ``dh_sf`` was invisible in the splitting. It is
    kept — now trivially satisfied — because the *reason* it was written still holds: a
    disagreement in the correction matrix only matters insofar as it reaches an observable,
    and that implication should be tested rather than assumed on the next occasion the two
    diverge.
    """
    s = record(key)["splitting_cm"]
    rel = abs(s["kuiva"] - s["plugin_pcc"]) / abs(s["kuiva"])
    assert rel < SPLITTING_TOL_RELATIVE, "{}: {:.2f} vs {:.2f} cm^-1".format(
        key, s["kuiva"], s["plugin_pcc"])


@pytest.mark.parametrize("key", KEYS)
def test_both_implementations_beat_the_uncorrected_hamiltonian_on_the_energy(key):
    """The energy-functional check, run against a second implementation of the same method.

    Asserted as a **ratio** rather than an absolute band, for the reason
    ``test_x2camf_dirac.py`` gives: a ratio is a statement about the correction and cannot be
    satisfied by a basis coincidence.
    """
    err = record(key)["energy_functional"]["error_eh"]
    none = abs(err["none"])
    for who in ("kuiva", "plugin_pcc"):
        assert none / abs(err[who]) > ENERGY_IMPROVEMENT_FACTOR, \
            "{} {}: {:+.3e} Eh against {:+.3e} with no correction".format(
                key, who, err[who], err["none"])


@pytest.mark.parametrize("key", KEYS)
def test_the_soc_only_variant_is_not_a_hamiltonian_correction(key):
    """⚠ The plugin's **default** entry point does nothing for the energy, and that is the
    point of recording it.

    ``x2camf.amfi`` without ``pcc=True`` returns only the spin-dependent part of the
    two-electron picture change. Its energy functional error is the *same as no correction at
    all* — measured +7.06e-03 Eh against +7.06e-03 for Ne — because the two-electron **scalar**
    picture change is the part that carries the energy, and it is the part Breit-Pauli AMFI and
    SNSO do not describe either .

    So this test asserts a **negative**: taking the plugin's default output for the whole
    correction would silently lose the larger half. It is the reason
    :data:`kuiva.amf.x2camf_plugin.VARIANTS` defaults to ``"pcc"``.
    """
    err = record(key)["energy_functional"]["error_eh"]
    improvement = abs(err["none"]) / abs(err["plugin_soc"])
    assert improvement < 3.0, \
        "{}: the spin-dependent-only variant improved the energy by {:.1f}x, which it should " \
        "not be able to do — has the plugin's amfi() started returning the scalar part " \
        "too?".format(key, improvement)


@pytest.mark.parametrize("key", KEYS)
def test_the_agreement_is_between_two_independent_implementations(key):
    """⚠ **The guard that agreement to 1e-08 makes necessary.**

    Once two codes agree to machine precision, every comparison above is also satisfied by a
    record in which the "plugin" column was never produced by the plugin — a caching mistake, a
    generator refactor that fed Kuiva's own matrix into both slots, a variant silently falling
    back. That failure mode did not exist while the two differed by 5-10%, and it is the price
    of the decoupling-convention change.

    So the record is asserted to contain *distinguishable* evidence of two implementations: the
    plugin's ``soc`` variant, which Kuiva has no counterpart to and which disagrees with the
    ``pcc`` variant in ``dh_sf`` by 38-100%. A record in which both columns came from one
    source cannot contain it.

    ⚠ The disagreement is asserted as a **relative** difference and not as "``soc`` is
    smaller", because it is not: under Coulomb the spin-dependent-only variant has essentially
    no scalar part (1.0e-05 against 3.0e-02 for Ne), but with **Gaunt** its ``max |dh_sf|``
    comes out *larger* than the full correction's (8.0e-02 against 4.5e-02). Both are simply
    different quantities from ``pcc``, which is all this test needs.
    """
    r = record(key)
    rel = r["primitive_basis"]["kuiva_vs_plugin_soc"]["rel_dh_sf"]
    assert rel > 0.1, \
        ("{}: the plugin's spin-dependent-only variant agrees with the full correction to "
         "{:.2e} in dh_sf, which it should not be able to do. Either the plugin has changed, "
         "or both columns came from one source.".format(key, rel))
    assert r["plugin_version"], "{}: no plugin version recorded".format(key)


def test_the_comparison_was_run_on_the_same_reference_configuration():
    """The plugin takes no configuration input, so a record generated against Kuiva's f-block
    M(3+) default would be comparing two different states .

    Asserted on the stored record rather than trusted, because it is the kind of variable that
    is free to get wrong and expensive to notice.
    """
    for key, r in records().items():
        if r.get("status") != "ok":
            continue
        neutral = AtomicConfiguration.ground(r["element"]).canonical
        assert r["configuration"] == neutral, \
            "{}: recorded against {} where the plugin used {}".format(
                key, r["configuration"], neutral)


def test_the_correction_is_a_genuine_fraction_of_the_one_electron_error():
    """A guard against a test suite that passes because *both* Hamiltonians are wrong.

    The uncorrected splitting is 15-40% above the four-component one across this set; if that
    gap ever vanished, every agreement above would be trivially satisfied.
    """
    for key, r in records().items():
        if r.get("status") != "ok":
            continue
        s = r["splitting_cm"]
        gap = abs(s["one_electron_x2c"] - s["four_component"]) / s["four_component"]
        assert gap > 0.10, "{}: the one-electron error is only {:.1%}".format(key, gap)


# --- the live comparison ------------------------------------------------------------------

@needs_plugin
def test_the_live_comparison_reproduces_the_stored_one():
    """Ne in the project default basis, computed now, against what was committed.

    The cheapest case in the set (about a second) and the one that catches a change in either
    implementation. Everything else here reads the file.
    """
    key = "Ne/x2c-SVPall-2c/coulomb"
    stored = record(key)
    mol = gto.M(atom="Ne 0 0 0", basis="x2c-SVPall-2c", verbose=0)
    configuration = AtomicConfiguration.ground("Ne")
    kuiva = amf_correction(mol, method="x2camf", interaction="coulomb",
                           configuration=configuration)
    h_sf, w = plugin.plugin_correction(mol, interaction="coulomb", variant="pcc",
                                       configuration=configuration)
    rel_w = np.max(np.abs(kuiva.w - w)) / max(np.max(np.abs(w)), np.max(np.abs(kuiva.w)))
    # ⚠ An order-of-magnitude comparison, not ``approx``. At 1e-08 the stored number is the
    # last few bits of a cancellation, and it is not reproducible across BLAS versions,
    # thread counts or compilers — pinning it to 0.1% would make this test fail on a different
    # machine for no physical reason. What is reproducible, and what the test is for, is that
    # the two implementations agree at all.
    stored_rel = stored["molecular_basis"]["kuiva_vs_plugin_pcc"]["rel_dw"]
    assert rel_w < CORRECTION_TOL_RELATIVE
    assert rel_w < 100.0 * max(stored_rel, 1e-12), \
        "live rel dw = {:.2e} against a stored {:.2e}".format(rel_w, stored_rel)


@needs_plugin
def test_the_external_method_and_the_direct_call_give_the_same_correction():
    """``method="x2camf-external"`` must be the plugin and nothing else.

    A wrapper that quietly did something extra — a factor, a projection, a different variant —
    is exactly what a like-for-like comparison forbids, so the two routes are compared rather
    than assumed equal.
    """
    mol = gto.M(atom="Ne 0 0 0", basis="x2c-SVPall-2c", verbose=0)
    external = amf_correction(mol, method="x2camf-external", interaction="coulomb")
    h_sf, w = plugin.plugin_correction(mol, interaction="coulomb", variant="pcc")
    assert np.allclose(external.h_sf, h_sf, atol=0, rtol=0)
    assert np.allclose(external.w, w, atol=0, rtol=0)
    assert external.method == "x2camf-external"
    assert external.backend == "x2camf-plugin"
    assert external.elements == ("Ne",)
    # The provenance must say what the plugin used, not what was asked for.
    assert external.configurations == {"Ne": AtomicConfiguration.ground("Ne").canonical}


@needs_plugin
def test_the_plugin_correction_is_time_reversal_even():
    """The structural invariant, on a matrix this project did not produce.

    ``validate_correction`` runs inside ``amf_correction`` and would have raised, so this
    asserts the *size* of what was projected out: an external matrix that needed a real
    projection would be a convention problem, not a rounding one.
    """
    mol = gto.M(atom="Ar 0 0 0", basis="x2c-SVPall-2c", verbose=0)
    external = amf_correction(mol, method="x2camf-external", interaction="coulomb")
    assert external.tr_residual_rel < 1e-10


@needs_plugin
def test_the_primitive_bases_of_the_two_codes_are_checked_not_assumed():
    """The plugin decontracts with ``mole.uncontracted_basis`` and Kuiva with
    ``decontract_basis(aggregate=True)``. They agree — and the code says so by checking the
    **overlap matrices**, because an AO ordering difference passes every shape check and
    produces a Hermitian correction of plausible magnitude over the wrong functions.
    """
    for basis in ("x2c-SVPall-2c", "dyallv2z", "ano-rcc-vdzp"):
        mol = gto.M(atom="Ne 0 0 0", basis=basis, verbose=0)
        xmol, contraction = plugin._decontracted(mol)
        assert contraction.shape == (int(xmol.nao), int(mol.nao))
        expected = mole.uncontracted_basis(gto.basis.load(basis, "Ne"))
        assert int(xmol.nao) == int(gto.M(atom="Ne 0 0 0", basis={"Ne": expected},
                                          verbose=0).nao)


@needs_plugin
def test_the_plugins_non_convergence_is_surfaced_as_a_warning(kuiva_caplog):
    """⚠ **The plugin reports failure by printing to stdout and returning anyway.**

    ``x2c2ePCC`` prints ``"SCF did not converge. x2c2ePCC cannot be used!"`` and then hands
    back the matrix — the ``exit(99)`` beside that message is commented out upstream — so from
    Python a failed run and a clean one are identical. Neutral titanium in the project default
    basis does exactly this, which is what makes it the test case.

    A **warning** rather than a refusal, and the evidence is why: the resulting ``max |dw|``
    is 8.384e-03 Eh against the 8.389e-03 Kuiva gets from its own converged solve, i.e. the
    number is fine and only the message is alarming. Refusing would throw away a working
    bisection tool over a threshold Kuiva does not control; not reporting it at all would be
    the silently-poor-correction failure this project keeps finding.
    """
    mol = gto.M(atom="Ti 0 0 0", basis="x2c-SVPall-2c", spin=2, verbose=0)
    correction = amf_correction(mol, method="x2camf-external", interaction="coulomb")
    assert correction.spin_orbit_scale == pytest.approx(8.384e-03, rel=1e-2)
    warnings = [r for r in kuiva_caplog.records if r.levelname == "WARNING"]
    assert any("did not converge" in r.getMessage() for r in warnings), \
        "the plugin printed a non-convergence diagnostic and nothing warned about it"


@needs_plugin
def test_the_plugin_does_not_write_into_the_output_stream(capfd):
    """INFO *is* the output file, so a C++ library printing into it would land
    unmarked in the middle of a formatted table.

    The plugin writes ``"Initializing 4c-HF for NE atom."`` and its total energy to
    ``std::cout`` on every call. That is captured at the file-descriptor level and routed to
    DEBUG, where diagnostics belong.
    """
    capfd.readouterr()
    mol = gto.M(atom="Ne 0 0 0", basis="x2c-SVPall-2c", verbose=0)
    amf_correction(mol, method="x2camf-external", interaction="coulomb")
    captured = capfd.readouterr()
    assert "Initializing 4c-HF" not in captured.out
    assert "DHF energy" not in captured.out


# --- what the plugin cannot do, refused rather than absorbed ------------------------------

@needs_plugin
def test_a_non_neutral_reference_configuration_is_refused():
    """⚠ The plugin has **no configuration input at all** — it takes an atomic number, a shell
    list and exponents. Kuiva's f-block default is the trivalent ion, so a comparison that let
    this through would silently compare Ce(3+) against neutral Ce.
    """
    mol = gto.M(atom="Ti 0 0 0", basis="x2c-SVPall-2c", verbose=0)
    with pytest.raises(NotImplementedError, match="neutral atom"):
        plugin.plugin_correction(mol, configuration="+3")
    with pytest.raises(NotImplementedError, match="neutral atom"):
        amf_correction(mol, method="x2camf-external", configuration="+3")


@needs_plugin
def test_decoupling_in_a_contracted_basis_is_refused():
    """``uncontract=False`` has no counterpart in the plugin, which always decontracts.

    Refused rather than ignored: silently returning the primitive-basis answer to a caller who
    asked for the contracted one is how the two would end up compared across a basis change,
    which is the named silently-poor-correction trap.
    """
    mol = gto.M(atom="Ne 0 0 0", basis="x2c-SVPall-2c", verbose=0)
    with pytest.raises(NotImplementedError, match="primitive basis"):
        plugin.plugin_correction(mol, uncontract=False)
    with pytest.raises(NotImplementedError, match="primitive basis"):
        amf_correction(mol, method="x2camf-external", uncontract=False)


def test_the_external_method_is_registered_but_is_not_the_default():
    """It is a bisection tool, not a fallback: nothing may select it on Kuiva's behalf.

    ⚠ The default is now ``"x2camf"``, so this asserts what it always meant
    rather than what it used to say: whatever the default is, it is **Kuiva's own**
    implementation and never the optional external one. A plugin that silently became the
    default would make every calculation depend on an optional build, and would quietly
    reintroduce its two limitations — neutral atoms only, and a hard ``exit(99)`` on cerium.
    """
    assert "x2camf-external" in METHODS
    mol = gto.M(atom="Ne 0 0 0", basis="x2c-SVPall-2c", verbose=0)
    assert amf_correction(mol).method == "x2camf"


def test_the_plugin_is_optional():
    """The whole point of the import gate: nothing in ``kuiva`` may import it eagerly.

    ``kuiva.amf`` is imported by the front-end, so an accidental top-level ``import x2camf``
    anywhere under it would turn a reference-only dependency into a runtime one.
    """
    import sys

    import kuiva.amf                                                        # noqa: F401
    assert "x2camf" not in sys.modules or plugin.available()
    assert plugin.version() == "" or plugin.available()
