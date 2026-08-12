"""Molecular assembly of the atomic mean-field correction.

Everything before this file was an atom. This one is about the two things that can go wrong
once there is more than one of them, and neither is about physics:

* **A block can land at the wrong AO offset.** The result is Hermitian, time-reversal even, of
  exactly the right magnitude, and wrong — the failure mode the mean field and
  :mod:`kuiva.amf.x2camf_plugin` both exist to prevent. It is checked three ways: against the
  isolated atom's own overlap matrix, by counting the non-zeros outside the diagonal blocks
  **exactly**, and — where the plugin is installed — against a second implementation of the
  same placement written by a different group.
* **A correction can be applied twice, or to the wrong Hamiltonian.** The record on
  :class:`~kuiva.interface.pyscf_bridge.SpinOrbitX2C` says what a Hamiltonian already
  contains, and the front-end is the one place that adds it.

The **cost** rule this file is written to: the fast tests use H, He, F, O, Li
and Cl, whose four-component atomic solves are seconds — the whole default file is about ten of
them. Titanium and anything heavier is ``slow`` or comes from the committed record. A molecular
test is never a good place to discover that an atomic solve is minutes long.
"""
import json

import numpy as np
import pytest
from pyscf import gto

from kuiva.amf import amf_correction
from kuiva.amf import x2camf_plugin as plugin
from kuiva.amf.atomic import cache_statistics, clear_cache
from kuiva.amf.correction import (ScreeningRecord, _check_atom_ordering,
                                  _check_off_atom_blocks_are_zero, _configuration_map,
                                  correction_memory_gb)
from kuiva.integrals.transform import transform_1e
from kuiva.interface.pyscf_bridge import ingest_spin_orbit
from kuiva.orth.canonical import canonical_orthogonalization
from kuiva.spinor.expand import expand_scalar_mos

BASIS = "x2c-SVPall-2c"

needs_plugin = pytest.mark.skipif(not plugin.available(),
                                  reason="the x2camf plugin is not installed "
                                         "(an optional external plugin)")


def molecule(atom, basis=BASIS, **kw):
    return gto.M(atom=atom, basis=basis, verbose=0, **kw)


def blocks(mol):
    """``[(atom index, p0, p1)]`` — the AO range of every atom."""
    slices = mol.aoslice_by_atom()
    return [(ia, int(slices[ia][2]), int(slices[ia][3])) for ia in range(mol.natm)]


def off_atom_nonzeros(mol, correction):
    """How many elements of the correction sit outside the atom-diagonal blocks."""
    inside = sum(int(np.count_nonzero(correction.h_sf[p0:p1, p0:p1]))
                 + int(np.count_nonzero(correction.w[:, p0:p1, p0:p1]))
                 for _, p0, p1 in blocks(mol))
    return (int(np.count_nonzero(correction.h_sf))
            + int(np.count_nonzero(correction.w)) - inside)


@pytest.fixture(scope="module")
def hf_molecule():
    """HF: one element with a real correction, one with none, and both cheap."""
    clear_cache()
    mol = molecule("H 0 0 0; F 0 0 0.917")
    return mol, amf_correction(mol, method="x2camf")


# --- Block placement -----------------------------------------------------------------------

def test_off_atom_blocks_are_exactly_zero(hf_molecule):
    """Atom-diagonality, asserted rather than assumed — and asserted **exactly**.

    ``== 0`` and not ``allclose``: off-atom blocks are never *computed*, only never written,
    so any non-zero there means a block landed at the wrong offset. A tolerance would let a
    small misplacement through, and a misplaced block is not small in its consequences.
    """
    mol, correction = hf_molecule
    assert off_atom_nonzeros(mol, correction) == 0
    # ...and there is something on the diagonal, or the test above is vacuous.
    assert correction.spin_orbit_scale > 0.0


def test_each_diagonal_block_is_the_isolated_atom_s_own_correction(hf_molecule):
    """The correction is atom-diagonal *by construction*, so a molecular block must be the
    isolated atom's block bit for bit — same cache entry, same arrays, no interpolation."""
    mol, correction = hf_molecule
    atomic = amf_correction(molecule("F 0 0 0", spin=1), method="x2camf")
    (_, _, _), (_, p0, p1) = blocks(mol)
    assert np.array_equal(correction.h_sf[p0:p1, p0:p1], atomic.h_sf)
    assert np.array_equal(correction.w[:, p0:p1, p0:p1], atomic.w)


@pytest.mark.stage_under_test("amf_atomic")   # asserts a SOLVE COUNT: no cache may serve it
def test_a_one_electron_element_contributes_exactly_nothing(hf_molecule):
    """Hydrogen has no second electron to screen, so its block is zero **by definition** —
    the same statement :func:`kuiva.amf.correction.amf_correction` makes for a one-electron
    molecule, made per element because a molecule containing hydrogen never reaches that
    branch. Computing it instead would picture-change a Hartree-Fock self-interaction.

    It must also cost nothing: a zero block that ran a four-component SCF would be a waste
    that no assertion on the numbers could see.
    """
    mol, correction = hf_molecule
    (_, h0, h1), _ = blocks(mol)
    assert not correction.h_sf[h0:h1, h0:h1].any()
    assert not correction.w[:, h0:h1, h0:h1].any()
    assert cache_statistics()["solves"] == 1                     # F only, never H


def test_the_ao_ordering_check_fires_on_a_permuted_block():
    """A validator that has never been observed to fail is not known to work — and this one
    guards the failure that is invisible to every norm-based test: a block whose AOs are in a
    different order is Hermitian, has the right eigenvalues, and is wrong.
    """
    mol = molecule("F 0 0 0", spin=1)
    s = mol.intor("int1e_ovlp")
    element = (mol.atom_pure_symbol(0), mol._basis["F"])
    _check_atom_ordering(element, s, "F")                        # the honest block passes

    perm = np.arange(mol.nao)
    perm[0], perm[-1] = perm[-1], perm[0]
    with pytest.raises(RuntimeError, match="ordering or normalization"):
        _check_atom_ordering(element, s[np.ix_(perm, perm)], "F")


def test_the_off_atom_check_fires_on_a_misplaced_block():
    """The other validator, broken the same way: a block written at the wrong offset."""
    n = 6
    h = np.zeros((n, n))
    w = np.zeros((3, n, n))
    ranges = [(0, 3), (3, 6)]
    h[0, 1] = h[1, 0] = 1.0
    _check_off_atom_blocks_are_zero(h, w, ranges)                # inside a block: fine

    h[0, 4] = h[4, 0] = 1.0                                      # spanning two atoms
    with pytest.raises(RuntimeError, match="outside the atom-diagonal blocks"):
        _check_off_atom_blocks_are_zero(h, w, ranges)


# --- Caching: one solve per unique element, counted -----------------------------------------

@pytest.mark.stage_under_test("amf_atomic")   # asserts a SOLVE COUNT: no cache may serve it
def test_one_solve_per_unique_element_not_per_atom():
    """One four-component solve per unique element, verified by **call count** and never by
    timing: on this
    machine wall time is partly a measure of the thermal envelope.

    Water, not a metal dimer, because the property under test is a counter and the cheapest
    molecule that exhibits it is the right one to test it on (the cheapest-system rule).
    ``ti2cl6`` is the ``slow`` test further down.
    """
    clear_cache()
    mol = molecule("O 0 0 0; H 0 0 0.958; H 0.926 0 -0.240")
    correction = amf_correction(mol, method="x2camf")
    assert cache_statistics()["solves"] == 1                     # O once; H is free
    assert correction.elements == ("O", "H")
    # The two hydrogens are the same element and get the same (zero) block; the oxygen block
    # is the only thing in the matrix.
    assert off_atom_nonzeros(mol, correction) == 0


@pytest.mark.stage_under_test("amf_atomic")   # asserts a SOLVE COUNT: no cache may serve it
def test_labelled_atoms_of_one_element_with_different_bases_stay_separate():
    """``mol.atom_symbol`` is the key, not the element: ``He1`` and ``He2`` carrying different
    bases are genuinely different atomic problems, and aliasing them would compute one and
    silently use it for both — over the wrong number of functions, so it would at least fail
    loudly here, but over the *same* number it would not."""
    clear_cache()
    mol = gto.M(atom="He1 0 0 0; He2 0 0 2.0",
                basis={"He1": BASIS, "He2": "cc-pvtz"}, verbose=0)
    correction = amf_correction(mol, method="x2camf")
    assert cache_statistics()["solves"] == 2
    assert set(correction.elements) == {"He1", "He2"}
    (_, a0, a1), (_, b0, b1) = blocks(mol)
    assert a1 - a0 != b1 - b0                                    # genuinely different bases
    assert off_atom_nonzeros(mol, correction) == 0


# --- Reference configurations across a molecule ---------------------------------------------

def test_a_single_configuration_is_refused_for_a_heteronuclear_molecule():
    """``configuration="+3"`` on a metal complex almost always means "the metal is trivalent".
    Applied to every element it would strip three electrons off each ligand atom as well, and
    the result would be a plausible-looking correction over the wrong reference states."""
    mol = molecule("Ti 0 0 0; Cl 0 0 2.3", spin=1)
    with pytest.raises(ValueError, match="cannot be applied to a molecule"):
        amf_correction(mol, method="x2camf", configuration="+3")


def test_configurations_are_mapped_per_element_and_a_stray_entry_is_refused():
    """The mapping logic, exercised without paying for a four-component solve.

    A mapping is looked up by atom label first and by element second, so ``{"Ti": ...}``
    covers ``Ti1`` and ``Ti2`` while ``{"Ti1": ...}`` can still separate them. An entry naming
    an element the molecule does not contain is an error, not a no-op: silently ignoring it
    would run the calculation with a different atomic reference from the one asked for.
    """
    labels = {"Ti1": ("Ti", "b"), "Ti2": ("Ti", "b"), "Cl": ("Cl", "b")}
    assert _configuration_map({"Ti": "+3"}, labels) == {
        "Ti1": "+3", "Ti2": "+3", "Cl": None}
    assert _configuration_map({"Ti1": "+4", "Ti": "+3"}, labels) == {
        "Ti1": "+4", "Ti2": "+3", "Cl": None}
    assert _configuration_map(None, labels) == {"Ti1": None, "Ti2": None, "Cl": None}
    with pytest.raises(ValueError, match="does not contain"):
        _configuration_map({"Fe": "+3"}, labels)


def test_a_per_element_configuration_reaches_the_atomic_solve():
    """End to end, on the cheapest molecule that can show it: He(+1) is a one-electron
    reference, so asking for it turns a non-zero block into an exactly zero one."""
    clear_cache()
    mol = molecule("He 0 0 0; He 0 0 2.0")
    neutral = amf_correction(mol, method="x2camf")
    ionic = amf_correction(mol, method="x2camf", configuration={"He": "+1"})
    assert neutral.spin_free_scale > 0.0
    assert ionic.is_zero
    assert ionic.configurations == {"He": "s1"}


# --- Multi-element behaviour -----------------------------------------------------------------

@pytest.mark.stage_under_test("amf_atomic")   # asserts a SOLVE COUNT: no cache may serve it
def test_the_correction_grows_steeply_with_z_across_a_molecule():
    """Wide-Z sanity: a molecule spanning a wide ``Z`` range behaves sanely.

    LiCl, Z = 3 against 17. The spin-orbit picture change is a core effect and scales steeply
    with nuclear charge, so the chlorine block must dominate the lithium one by orders of
    magnitude — the check that would catch a block placed on the wrong *atom* even when both
    blocks happen to be the same size.
    """
    clear_cache()
    mol = molecule("Li 0 0 0; Cl 0 0 2.021")
    correction = amf_correction(mol, method="x2camf")
    (_, li0, li1), (_, cl0, cl1) = blocks(mol)
    li = float(np.max(np.abs(correction.w[:, li0:li1, li0:li1])))
    cl = float(np.max(np.abs(correction.w[:, cl0:cl1, cl0:cl1])))
    assert li > 0.0 and cl > 0.0                                 # both were computed
    assert cl > 100.0 * li                                       # measured: ~1.6e3x
    assert cache_statistics()["solves"] == 2


def test_kramers_degeneracy_is_exact_with_the_correction():
    """The structural guarantee must survive assembly. It does so *by construction* — ``dh_sf`` is
    real symmetric and ``dw`` real antisymmetric out of ``decompose_two_component``, so the
    assembled correction is time-reversal even and cannot split a Kramers pair — but that is a
    claim about the code path and not about the arithmetic, so it is measured.

    The tolerance is 1e-12 Eh, four orders inside the 1e-8 Eh band reserved for genuine
    numerical Kramers splitting.
    """
    mol = molecule("H 0 0 0; F 0 0 0.917")
    h = ingest_spin_orbit(mol, screening="x2camf").hamiltonian()
    ob = canonical_orthogonalization(mol.intor("int1e_ovlp"))
    sb = expand_scalar_mos(ob.x).transform_scalar_basis(ob.x, "ao")
    ev = np.sort(np.linalg.eigvalsh(transform_1e(h, sb.c)))
    assert np.max(np.abs(ev[0::2] - ev[1::2])) < 1e-12


# --- The front-end seam ----------------------------------------------------------------------

def test_screening_none_is_bitwise_the_bare_one_electron_operator():
    """``screening="none"`` must be the untouched X2C Hamiltonian, to the bit.

    ⚠ This test changed shape when X2CAMF became the default. It used to
    compare the *default* against an explicit ``"none"``, which was the statement that the
    correction could land without disturbing anything. That comparison is now vacuous — the
    default is the correction — so what is asserted instead is the thing that still needs to be
    true and is no longer implied by anything else: the escape hatch is a genuine escape hatch,
    reproducing PySCF's two-component ``get_hcore`` exactly, and not merely a smaller
    correction.

    ``array_equal`` and not ``allclose``: 1 ulp on a heavy-element matrix element is not
    nothing, and there is no arithmetic between the two sides that could legitimately produce
    one.
    """
    from pyscf.x2c import x2c
    from kuiva.spinor.expand import decompose_two_component

    mol = molecule("H 0 0 0; F 0 0 0.917")
    plain = ingest_spin_orbit(mol, screening="none")

    helper = x2c.SpinOrbitalX2CHelper(mol)
    helper.approx = "1e"
    h_sf, w = decompose_two_component(np.asarray(helper.get_hcore()))

    assert np.array_equal(plain.h_sf, h_sf)
    assert np.array_equal(plain.w, w)
    assert plain.screening.method == "none" and not plain.screening.applied
    assert plain.screening.elements == ()


def test_the_correction_is_added_before_any_change_of_basis(hf_molecule):
    """⚠ *Where* the correction is added is a design decision, not an implementation detail.

    Adding it in the AO basis is what keeps :meth:`SpinOrbitX2C.transform` working untouched:
    ``dh_sf`` and ``dw`` transform under a change of scalar basis exactly as the one-electron
    parts do, so once summed in there is nothing left to transform separately and no way for a
    caller to transform one and forget the other. Asserted by doing it in both orders and
    demanding they agree to rounding.

    The tolerance is absolute and relative to the *operator*, not to each element: summing
    before and after a congruence differ only by the non-associativity of floating-point
    addition, which is bounded by ulps of the largest element (order 40 Eh here), not of the
    element being compared. Observed worst difference 1.3e-13 Eh, against the 1e-8 Eh output precision.
    """
    mol, correction = hf_molecule
    ob = canonical_orthogonalization(mol.intor("int1e_ovlp"))
    added_then_transformed = ingest_spin_orbit(mol, screening="x2camf").transform(ob.x)
    transformed_then_added = ingest_spin_orbit(mol, screening="none").transform(ob.x)
    expected_h = transformed_then_added.h_sf + ob.x.T @ correction.h_sf @ ob.x
    expected_w = transformed_then_added.w + np.stack(
        [ob.x.T @ wk @ ob.x for wk in correction.w])
    assert np.max(np.abs(added_then_transformed.h_sf - expected_h)) < 1e-11
    assert np.max(np.abs(added_then_transformed.w - expected_w)) < 1e-11


def test_the_hamiltonian_records_what_it_contains(hf_molecule):
    """The provenance-record deliverables, which land here because they only mean anything once
    a driver applies the correction. A stored property matrix that does not say whether it was
    screened is not interpretable — the difference is 15-30% on every splitting in it."""
    mol, _ = hf_molecule
    soc = ingest_spin_orbit(mol, screening="x2camf", interaction="gaunt")
    record = soc.screening
    assert isinstance(record, ScreeningRecord) and record.applied
    assert record.method == "x2camf" and record.interaction == "gaunt"
    assert record.backend == "pyscf" and record.backend_version
    assert set(record.elements) == {"H", "F"}
    assert record.configurations["F"] == "s4 p5"                 # the neutral atom
    assert record.spin_orbit_scale > 0.0
    # The dict form is the contract with stored data: it must survive a json round trip.
    dumped = json.loads(json.dumps(soc.provenance()))
    assert dumped["screening"]["method"] == "x2camf"
    assert dumped["screening"]["configurations"]["F"] == "s4 p5"
    # ...and it survives a change of basis, because the Hamiltonian it describes does.
    ob = canonical_orthogonalization(mol.intor("int1e_ovlp"))
    assert soc.transform(ob.x).screening == record


def test_the_atomic_x_asymmetry_is_negligible_against_the_correction(hf_molecule):
    """⚠ The recorded asymmetry, measured — **on a molecule, because an atom cannot
    show it**.

    Kuiva's one-electron path uses exact *molecular* X2C (``approx="1e"``) while the AMF
    ``X``/``R`` are atomic by construction. That is standard for X2CAMF and is documented
    rather than hidden, and it went unmeasured for a while, because on a single atom the
    two decouplings coincide to 1e-6 by construction (``test_soc_ingestion.py``).

    Setting ``approx="atom1e"`` makes them consistent. Measured: it moves the corrected
    spin-orbit operator by 2.2e-5 (HF) and 5.1e-7 (LiCl) relative, against corrections of 24%
    and 12% of that operator — four to six orders below the thing being corrected. The
    assertion is on the **ratio**, not on either number alone: what matters is not that the
    asymmetry is small but that it is small *compared to the correction*.
    """
    mol, _ = hf_molecule
    consistent = ingest_spin_orbit(mol, approx="atom1e", screening="x2camf")
    mixed = ingest_spin_orbit(mol, approx="1e", screening="x2camf")
    uncorrected = ingest_spin_orbit(mol, approx="1e", screening="none")

    scale = float(np.max(np.abs(mixed.w)))
    asymmetry = float(np.max(np.abs(mixed.w - consistent.w))) / scale
    correction = float(np.max(np.abs(mixed.w - uncorrected.w))) / scale
    assert correction > 0.05                                     # the correction is real
    assert asymmetry < 1e-3 * correction                         # and the asymmetry is not


def test_the_correction_reduces_the_molecular_spin_orbit_operator(hf_molecule):
    """Screening always reduces the spin-orbit coupling, and by a few tens of per cent. A
    correction of a few per mille or of a factor of two would both mean something is wrong.
    Measured here: 23.6% on HF, matching the ~23% recorded for neon in ``test_amf_correction``.
    """
    mol, _ = hf_molecule
    plain = ingest_spin_orbit(mol, screening="none")
    screened = ingest_spin_orbit(mol, screening="x2camf")
    reduction = 1.0 - screened.soc_strength / plain.soc_strength
    assert 0.10 < reduction < 0.40


# --- resource accounting ---------------------------------------------------------------------------

def test_the_correction_appears_in_the_memory_plan_only_when_it_is_asked_for():
    """A new large array gets an exact sizing function and a place in the pre-flight.

    The sizing function itself is pinned two-sidedly against a real array's ``nbytes`` in
    ``test_resources.py``, with every other one. What is checked here is that the phase is
    conditional — a plan that always carried it would over-state the requirement of every
    unscreened calculation, and the memory limit is a hard error, so pessimism refuses runs.
    """
    from kuiva.interface.pyscf_bridge import memory_plan

    plain = [p.name for p in memory_plan(40)]
    screened = [p.name for p in memory_plan(40, screening=True)]
    assert "two-electron SOC picture change" not in plain
    assert "two-electron SOC picture change" in screened
    phase, = [p for p in memory_plan(40, screening=True)
              if p.name == "two-electron SOC picture change"]
    assert phase.allocations[0].gb == pytest.approx(correction_memory_gb(40), rel=1e-12)


# --- The plugin: a second implementation of the same block placement ---------------------------

@needs_plugin
def test_the_plugin_places_the_same_blocks_in_the_same_positions():
    """⚠ **This is the only test in the file that is not about Kuiva's opinion of Kuiva.**

    ``x2camf.amfi`` does its own molecular assembly — one atomic block per unique element,
    placed over ``xmol.aoslice_2c_by_atom()`` — so it is a second implementation of exactly
    this placement, written by the group that published the method, in a different basis
    convention. "Off-atom blocks are exactly zero" and "the blocks landed on the right AOs"
    therefore stop being assertions about our own code.

    The residual disagreement is the known ``X``-convention difference
    (Fock-derived against one-electron), which is 1e-4-ish on ``dw`` in a contracted molecular
    basis and does not move a splitting. Both sides are pinned to the **neutral** reference,
    because the plugin has no configuration input at all.
    """
    clear_cache()
    mol = molecule("H 0 0 0; F 0 0 0.917")
    ours = amf_correction(mol, method="x2camf")
    theirs = amf_correction(mol, method="x2camf-external")

    # Structure first: the same zeros in the same places, exactly.
    assert off_atom_nonzeros(mol, theirs) == 0
    (_, h0, h1), _ = blocks(mol)
    assert not theirs.h_sf[h0:h1, h0:h1].any()                   # hydrogen, from both codes

    # Then the numbers, block by block, so a placement error cannot average away.
    for _, p0, p1 in blocks(mol)[1:]:
        scale = float(np.max(np.abs(ours.w[:, p0:p1, p0:p1])))
        assert scale > 0.0
        assert np.max(np.abs(ours.w[:, p0:p1, p0:p1]
                             - theirs.w[:, p0:p1, p0:p1])) < 5e-3 * scale
        sf = float(np.max(np.abs(ours.h_sf[p0:p1, p0:p1])))
        assert np.max(np.abs(ours.h_sf[p0:p1, p0:p1]
                             - theirs.h_sf[p0:p1, p0:p1])) < 2e-3 * sf


@needs_plugin
def test_the_plugin_refuses_a_configuration_it_cannot_express_anywhere_in_the_molecule():
    """The guard has to look at **every** element, not at atom 0. A molecule whose ligands are
    neutral and whose metal is trivalent would otherwise pass a check on the first atom and
    compare two different reference states."""
    mol = molecule("H 0 0 0; F 0 0 0.917")
    with pytest.raises(NotImplementedError, match="takes no configuration input"):
        amf_correction(mol, method="x2camf-external", configuration={"F": "+1"})


# --- The committed end-to-end record -----------------------------------------------------------

def stored_systems():
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "tests/reference/amf_molecular.json"
    if not path.is_file():
        pytest.skip("amf_molecular.json not generated (tests/generate/amf_molecular.py)")
    return json.loads(path.read_text())["systems"]


@pytest.mark.parametrize("key", ["ne", "ticl3", "ti2cl6", "hi", "cecl3"])
def test_the_recorded_end_to_end_runs_hold(key):
    """The molecular exit criteria, asserted against the committed record rather than re-run.

    The recorded systems span an atom, a metal-with-ligands, a dimer with repeated atoms
    of *both* elements, a heavy p-block diatomic and an f-block complex — 40 minutes of
    four-component atomic solves between them, generated once (the bounded-run rule requires the
    user's approval for exactly that, and it was given). What is asserted is what the
    generator cannot get wrong by accident: exact structural zeros, exact solve counts, and
    bands on the physics.
    """
    systems = stored_systems()
    if key not in systems:
        pytest.skip("{} not in the committed record".format(key))
    record = systems[key]

    assert record["off_atom_nonzeros"] == 0                      # exact, not a tolerance
    # ⚠ **One acquisition per unique element, never per atom** — asserted on solves *plus*
    # disk hits, not on solves alone. Since the correction gained a persistent cache
    # a generator run on a machine that already has an element cached records
    # a *hit* where a fresh clone records a *solve*, and both are correct. What must never
    # happen is Ti2Cl6 acquiring eight blocks for two elements, and the sum catches that
    # exactly. An element whose reference has one electron or none contributes nothing and is
    # subtracted; hydrogen never reaches the solver at all.
    expected = len(record["elements"]) - sum(
        1 for e in record["per_element"].values() if e["dw"] == 0.0)
    acquired = record["solves"] + record.get("disk_hits", 0)
    assert acquired == expected, (
        "{}: {} solves + {} disk hits for {} correctable elements".format(
            key, record["solves"], record.get("disk_hits", 0), expected))
    assert record["scf_converged"]
    # The two halves of the correction, reported separately and never summed: the
    # spin-free part is the larger one, which is the feature that distinguishes this method
    # from Breit-Pauli AMFI and SNSO.
    assert record["spin_free_scale"] > record["spin_orbit_scale"]
    # Screening always reduces the spin-orbit operator. ⚠ The band is wide and the quantity is
    # a **diagnostic, not a splitting**: ``max |w|`` is a core matrix element of the heaviest
    # element, where the fractional screening is smallest, so it falls from 21.7% (Ne) to 4.1%
    # (HI) across the set without saying anything about the valence fine structure. The number
    # that means something is the same-basis four-component j-splitting.
    assert 0.02 < record["soc_reduction"] < 0.40
    # the ingestion rule's structural guarantee, on a molecule.
    assert record["tr_residual_rel"] < 1e-6


def test_every_recorded_plugin_comparison_agrees_on_the_block_placement():
    """⚠ The cross-code half of the record: where the plugin is comparable it must place the
    same blocks in the same positions, with the residual being the known ``X``-convention
    difference and nothing else.

    ``cecl3`` is deliberately *not* comparable — Kuiva's f-block default is Ce(3+) and the
    plugin has no configuration input — and the record says so rather than quietly comparing
    two different calculations.
    """
    compared = 0
    for key, record in stored_systems().items():
        comparison = record.get("plugin")
        if comparison is None:
            continue
        if not comparison["comparable"]:
            assert comparison["non_neutral_reference"]           # and it says which
            continue
        assert comparison["off_atom_nonzeros"] == 0
        # ⚠ **Both halves now agree to ~1e-6 or better, and this file used to assert a band
        # around 1e-4...1e-2 on the spin-free half.** The band was real: the two codes
        # decoupled with a different ``X``, and the residue grew with Z (2.7e-4 Ne, 6.1e-4
        # TiCl3, 3.7e-3 HI) exactly as its explanation predicted, because the difference lives
        # in the high-exponent primitive space and the heavier the element the more of that
        # space survives the contraction back to the molecular basis. Kuiva adopted the
        # plugin's convention, so what is left is arithmetic order.
        #
        # The bound is one-sided now because there is no longer a difference whose
        # disappearance would be suspicious — but it is set at 1e-4, three orders below the
        # old band's ceiling and two above the observed 1.6e-6, so a *return* of the
        # convention difference would still fail it.
        assert comparison["dw_relative"] < 1e-4, key
        assert comparison["dh_sf_relative"] < 1e-4, key
        compared += 1
    assert compared >= 2                                         # the record is not vacuous


# --- The recorded systems, at their real cost --------------------------------------------

@pytest.mark.slow
@pytest.mark.stage_under_test("amf_atomic")   # asserts a SOLVE COUNT: no cache may serve it
def test_ti2cl6_solves_once_per_element():
    """The system for the per-element solve count: 2 Ti and 6 Cl, **2 solves**.

    ``slow`` because neutral Ti and Cl are open-shell four-component solves and
    average-of-configuration costs an extra build per SCF cycle. The
    cheap version of this assertion is ``test_one_solve_per_unique_element_not_per_atom``;
    this one exists because the counter should be seen to hold on a real system with repeated
    atoms of *both* elements, which water does not have.
    """
    # ⚠ Imported through the repository root rather than as ``tests.generate.systems``, which
    # only resolves when pytest is invoked from the root. Running ``pytest`` from inside
    # ``tests/`` is a reasonable thing to do and used to fail this one test with a
    # ``ModuleNotFoundError`` three hours into the slow suite — a cwd-dependent test is a trap
    # regardless of which cwd is documented.
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests/generate"))
    from systems import SYSTEMS_BY_KEY

    system = SYSTEMS_BY_KEY["ti2cl6"]
    mol = gto.M(atom=[(s, xyz) for s, xyz in system.atoms], basis=system.basis,
                charge=system.charge, spin=system.spin, verbose=0)
    clear_cache()
    correction = amf_correction(mol, method="x2camf")
    assert cache_statistics()["solves"] == 2                     # not 8
    assert set(correction.elements) == {"Ti", "Cl"}
    assert off_atom_nonzeros(mol, correction) == 0

    # Every Ti block is the same block, and so is every Cl block: the correction depends on
    # the element and its basis and on nothing else about where the atom sits.
    per_element = {}
    for ia, p0, p1 in blocks(mol):
        per_element.setdefault(mol.atom_symbol(ia), []).append(correction.h_sf[p0:p1, p0:p1])
    for same in per_element.values():
        assert all(np.array_equal(same[0], other) for other in same[1:])
