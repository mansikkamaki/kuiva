"""The X2C method surface: named Hamiltonians, the two axes, and provenance.

Two things are being protected here, and neither is arithmetic.

**That a calculation cannot run with a Hamiltonian nobody asked for.** The surface has a name
knob *and* two axis knobs, and the failure mode of any such design is that a contradiction
between them gets quietly reconciled. It is refused instead, and so is a
``decoupling_options=`` that applies to a route the calculation is not taking.

**That a stored result says which of several Hamiltonians produced it.** :class:`SpinOrbitX2C`
now carries a :class:`~kuiva.x2c.methods.DecouplingRecord` beside its
:class:`~kuiva.amf.correction.ScreeningRecord`, both of which are contracts with stored data
. A property matrix that does not say whether it was screened *and* whether its decoupling
was local is not interpretable.
"""
import json

import numpy as np
import pytest

from kuiva.interface.pyscf_bridge import ingest_spin_orbit, local_x2c_hamiltonian
from kuiva.spinor.expand import time_reversal_residual
from kuiva.x2c.methods import (DEFAULT_METHOD, DecouplingRecord, known_methods, resolve)


def _mol(atom="Ti 0 0 0; Cl 0 0 2.2; Cl 1.905 0 -1.1; Cl -1.905 0 -1.1", **kw):
    from pyscf import gto
    return gto.M(atom=atom, basis=kw.pop("basis", "x2c-SVPall-2c"), spin=kw.pop("spin", None),
                 verbose=0, **kw)


# --- Resolution ---------------------------------------------------------------------------

def test_the_canonical_methods_resolve_to_the_right_axes():
    assert known_methods() == ("X2C-1e", "X2C-AMF", "X2C-1e-DLU", "X2C-AMF-DLU", "X2C-mmf")
    assert DEFAULT_METHOD == "X2C-AMF"
    # ⚠ Exactly one method is not for production, and it is the benchmark one. If this count
    # ever changes, either a benchmark method became a default or a production method quietly
    # stopped being one.
    assert [m for m in known_methods() if not resolve(m).production] == ["X2C-mmf"]
    expected = {"X2C-1e": ("1e", "none"), "X2C-AMF": ("1e", "x2camf"),
                "X2C-1e-DLU": ("1e-dlu", "none"), "X2C-AMF-DLU": ("1e-dlu", "x2camf"),
                "X2C-mmf": ("1e", "mmf")}
    for name, (decoupling, screening) in expected.items():
        method = resolve(name)
        assert (method.decoupling, method.screening) == (decoupling, screening)
        assert resolve(name.lower()) is method and resolve(name.upper()) is method


def test_axes_alone_find_the_canonical_name_and_the_defaults_are_amf():
    assert resolve(decoupling="1e", screening="x2camf").name == "X2C-AMF"
    assert resolve().name == "X2C-AMF"                       # both axes defaulted
    assert resolve(screening="none").name == "X2C-1e"
    assert resolve(decoupling="1e-dlu").name == "X2C-AMF-DLU"


def test_a_valid_but_unnamed_combination_is_synthesized_not_refused():
    """``atom1e`` is reachable and useful (it makes the one- and two-electron decouplings
    consistent) but has no canonical name. Provenance must still carry something actionable
    rather than an empty string."""
    method = resolve(decoupling="atom1e", screening="x2camf")
    assert method.decoupling == "atom1e" and method.screening == "x2camf"
    assert "atom1e" in method.name and method.name not in known_methods()


def test_a_name_contradicting_an_axis_is_refused():
    """⚠ The whole reason the surface is safe. Letting either side win silently would mean a
    calculation running with a Hamiltonian that was never requested."""
    with pytest.raises(ValueError, match="contradict"):
        resolve("X2C-AMF", screening="none")
    with pytest.raises(ValueError, match="contradict"):
        resolve("X2C-1e-DLU", decoupling="1e")
    # Agreeing is fine, and is not a contradiction.
    assert resolve("X2C-AMF", decoupling="1e", screening="x2camf").name == "X2C-AMF"


def test_unknown_names_and_axis_values_are_refused_with_the_alternatives():
    with pytest.raises(ValueError, match="unknown Hamiltonian method"):
        resolve("X2C-mmf-DLU")                # a plausible-looking name that does not exist
    with pytest.raises(ValueError, match="unknown decoupling"):
        resolve(decoupling="dlh")             # deliberately rejected
    with pytest.raises(ValueError, match="unknown screening"):
        resolve(screening="snso")             # a rejected alternative


# --- Provenance ---------------------------------------------------------------------------

def test_decoupling_record_round_trips_through_json():
    record = DecouplingRecord(decoupling="1e-dlu", implementation="kuiva", partition="atoms",
                              source="diagonal", fragments=4,
                              block_scales={"Ti": 5.93, "Cl": 1.01})
    restored = json.loads(json.dumps(record.as_dict()))
    assert restored["decoupling"] == "1e-dlu" and restored["fragments"] == 4
    assert restored["block_scales"]["Ti"] == pytest.approx(5.93)
    assert record.local and record.max_block_scale == pytest.approx(5.93)


def test_the_default_record_describes_the_exact_path():
    """``DecouplingRecord()`` must describe what every committed reference was produced with,
    so that an old stored record with no decoupling field reads correctly."""
    record = DecouplingRecord()
    assert record.decoupling == "1e" and not record.local
    assert record.as_dict()["implementation"] == "pyscf"


def test_partition_single_is_not_reported_as_local():
    """⚠ ``partition="single"`` is the *exact* transformation through the local code path. It
    must not be recorded as an approximation, or the like-for-like DLU reference would carry a
    warning saying it was approximate."""
    assert not DecouplingRecord(decoupling="1e-dlu", partition="single").local
    assert DecouplingRecord(decoupling="1e-dlu", partition="atoms").local


def test_ingestion_records_the_method_and_the_decoupling():
    mol = _mol("Ne 0 0 0")
    exact = ingest_spin_orbit(mol, screening="none")
    assert exact.method == "X2C-1e"
    assert exact.decoupling.decoupling == "1e" and not exact.decoupling.local
    provenance = exact.provenance()
    assert provenance["method"] == "X2C-1e"
    assert provenance["decoupling"]["implementation"] == "pyscf"
    json.dumps(provenance)                                   # the dump header must serialize

    local = ingest_spin_orbit(mol, approx="1e-dlu", screening="none")
    assert local.method == "X2C-1e-DLU"
    assert local.decoupling.local and local.decoupling.fragments == 1
    assert set(local.decoupling.block_scales) == {"Ne"}


def test_transform_carries_the_records_into_the_working_basis():
    """⚠ A change of basis must not lose provenance. The transformed operator is what reaches
    the property dump, so a record dropped here is a record that never gets written."""
    mol = _mol("Ne 0 0 0")
    soc = ingest_spin_orbit(mol, approx="1e-dlu", screening="none")
    moved = soc.transform(np.eye(soc.nao))
    assert moved.method == soc.method
    assert moved.decoupling == soc.decoupling
    assert moved.screening == soc.screening


def test_selecting_the_local_decoupling_warns(kuiva_caplog):
    """"Proceeds but the user should know". DLU is the cheap end of the ladder and a
    result produced with it must not be mistaken later for a standard one."""
    with kuiva_caplog.at_level("WARNING"):
        ingest_spin_orbit(_mol("Ne 0 0 0"), approx="1e-dlu", screening="none").report()
    assert any("LOCAL (DLU)" in r.message for r in kuiva_caplog.records)


# --- The routing itself -------------------------------------------------------------------

def test_decoupling_options_are_refused_on_the_exact_route():
    """⚠ Silently ignoring them would mean running a different Hamiltonian from the one
    requested — the same failure the name/axis contradiction check exists to prevent."""
    mol = _mol("Ne 0 0 0")
    with pytest.raises(ValueError, match="only apply to the local"):
        ingest_spin_orbit(mol, approx="1e", screening="none",
                          decoupling_options={"source": "isolated"})
    with pytest.raises(ValueError, match="unknown one-electron decoupling"):
        ingest_spin_orbit(mol, approx="dlh", screening="none")


def test_single_partition_reproduces_the_exact_transformation_through_the_local_path():
    """⚠ This is the like-for-like reference any DLU accuracy claim must use.

    Kuiva's exact decoupling and PySCF's differ by up to 2.4e-07 relative on a heavy element
, so comparing ``"1e"`` against ``"1e-dlu"`` measures the DLU approximation *plus*
    that difference. ``partition="single"`` removes the confound by construction: same code,
    one fragment.
    """
    from kuiva.x2c.decouple import decoupling_matrices, picture_change
    from kuiva.interface.pyscf_bridge import four_component_one_electron

    mol = _mol()
    h_single, record = local_x2c_hamiltonian(mol, partition="single")
    assert not record.local and record.fragments == 1

    fc = four_component_one_electron(mol)
    x, r = decoupling_matrices(fc.hcore, fc.overlap, fc.light_speed)
    assert np.array_equal(h_single, fc.contract(picture_change(fc.hcore, x, r)))

    # ...and it is measurably not PySCF's, which is exactly why it has to exist.
    h_pyscf = ingest_spin_orbit(mol, approx="1e", screening="none").hamiltonian()
    difference = np.max(np.abs(h_single - h_pyscf)) / np.max(np.abs(h_pyscf))
    assert 1e-12 < difference < 1e-6


def test_isolated_source_is_refused_for_a_single_partition():
    """The combination is meaningless — the "isolated fragment" would be the molecule — and a
    meaningless request is refused rather than reinterpreted."""
    with pytest.raises(ValueError, match="no meaning with partition='single'"):
        local_x2c_hamiltonian(_mol(), partition="single", source="isolated")
    with pytest.raises(ValueError, match="unknown partition"):
        local_x2c_hamiltonian(_mol("Ne 0 0 0"), partition="fragments")


def test_a_dlu_hamiltonian_keeps_the_structural_guarantees():
    """the ingestion rule's guarantees are about the *operator*, so they must survive whichever decoupling
    produced it: time-reversal even to machine precision, and a spin-orbit part that is real
    antisymmetric in the fixed convention."""
    mol = _mol()
    soc = ingest_spin_orbit(mol, approx="1e-dlu", screening="none")

    assert soc.tr_residual_rel < 1e-10
    _, relative = time_reversal_residual(soc.hamiltonian())
    assert relative < 1e-12
    assert np.max(np.abs(soc.h_sf - soc.h_sf.T)) / np.max(np.abs(soc.h_sf)) < 1e-14
    assert np.max(np.abs(soc.w + np.transpose(soc.w, (0, 2, 1)))) / soc.soc_strength < 1e-14


def test_the_two_axes_are_independent_to_machine_precision():
    """⚠ What lets the DLU characterization, measured entirely at ``screening="none"``, be
    quoted for ``X2C-AMF-DLU`` as well.

    The two-electron correction is built from atomic four-component solves and added in the AO
    basis; it knows nothing about which one-electron decoupling it is being added to. So the
    correction must come out **identical** on the exact and the local route, and the DLU error
    must therefore be the same with screening on as with it off.

    Not bitwise: ``(a + c) - (b + c)`` is not ``a - b`` in floating point, so the residual is
    ~2e-15 absolute on a 4.5e-03 correction (4e-13 relative). Asserting bitwise equality here
    would be asserting a property of the summation order, not of the physics.

    HF rather than a heavy molecule on purpose: one four-component solve, 0.7 s cold, so the
    fast suite pays for the statement rather than skipping it.
    """
    mol = _mol("H 0 0 0; F 0 0 0.917")
    hamiltonians = {}
    for approx in ("1e", "1e-dlu"):
        for screening in ("none", "x2camf"):
            hamiltonians[(approx, screening)] = ingest_spin_orbit(
                mol, approx=approx, screening=screening).hamiltonian()

    correction_exact = hamiltonians[("1e", "x2camf")] - hamiltonians[("1e", "none")]
    correction_local = hamiltonians[("1e-dlu", "x2camf")] - hamiltonians[("1e-dlu", "none")]
    assert float(np.max(np.abs(correction_exact))) > 1e-4, \
        "the correction must be non-trivial or this test is vacuous"

    # ⚠ **Every bound here is against the Hamiltonian's own scale**, which is a fixed physical
    # quantity, and never against another rounding-level residual. Two earlier versions of this
    # test were flaky for exactly that reason: normalizing by the DLU error (a small difference
    # of large numbers) put the ratio at 2e-11 against a 1e-11 bound, and bounding one residual
    # by a multiple of another fails outright whenever the second is exactly zero. Threaded BLAS
    # fixes no reduction order, so the last bits move between runs and a test whose verdict
    # depends on them is measuring the summation, not the physics.
    h_scale = float(np.max(np.abs(hamiltonians[("1e", "none")])))
    residual = float(np.max(np.abs(correction_local - correction_exact)))
    assert residual < 1e-13 * h_scale

    # The corollary, and it is the *same* quantity rearranged rather than a second measurement:
    # (h_dlu^amf - h_1e^amf) - (h_dlu^none - h_1e^none) == correction_local - correction_exact
    # in exact arithmetic. So the DLU error is the same with screening on as with it off, which
    # is what lets the screening="none" characterization be quoted for X2C-AMF-DLU.
    error_off = hamiltonians[("1e-dlu", "none")] - hamiltonians[("1e", "none")]
    error_on = hamiltonians[("1e-dlu", "x2camf")] - hamiltonians[("1e", "x2camf")]
    rearranged = float(np.max(np.abs((error_on - error_off)
                                     - (correction_local - correction_exact))))
    assert rearranged < 1e-13 * h_scale


def test_dlu_moves_the_spin_orbit_part_far_less_than_the_spin_free_one():
    """Measured on TiCl3, and recorded because it shapes what stage 4 has to measure: the DLU
    error lands almost entirely in the **spin-free** part (1.1e-04 Eh) and barely touches the
    spin-orbit operator (2.5e-07 Eh on a scale of 0.08, i.e. 3e-06 relative).

    ⚠ That is a statement about matrix elements, not about splittings — a spin-free shift can
    still move a state energy. It bounds nothing on its own and is not an accuracy claim.
    """
    mol = _mol()
    exact = ingest_spin_orbit(mol, approx="1e", screening="none")
    local = ingest_spin_orbit(mol, approx="1e-dlu", screening="none")

    spin_free = float(np.max(np.abs(exact.h_sf - local.h_sf)))
    spin_orbit = float(np.max(np.abs(exact.w - local.w)))
    assert spin_free < 1e-3
    assert spin_orbit < 1e-5
    assert spin_orbit < spin_free


# --- X2C-mmf: the molecular mean field -----------------------------------

def test_mmf_is_a_screening_value_and_cannot_combine_with_x2camf():
    """⚠ The design point: mmf and X2CAMF are the **same subtraction**, so they live on one
    axis and there is no combination to refuse. A design that removes a double-counting failure
    mode is worth more than one that guards against it."""
    from kuiva.x2c.methods import SCREENINGS

    assert "mmf" in SCREENINGS
    method = resolve("X2C-mmf")
    assert (method.decoupling, method.screening) == ("1e", "mmf")
    assert not method.production, "a benchmark method must declare itself"
    assert resolve(screening="mmf").name == "X2C-mmf"
    # Selecting one axis value excludes the other by construction.
    with pytest.raises(ValueError, match="contradict"):
        resolve("X2C-mmf", screening="x2camf")


def test_mmf_reproduces_x2camf_on_a_closed_shell_atom():
    """⚠ The test that transfers X2CAMF's entire validation chain to the molecular path.

    For an isolated closed-shell atom the two methods solve the **identical** four-component
    problem — X2CAMF's "atomic" solve *is* the molecular one — so the corrections must agree to
    numerical precision rather than merely closely. Measured on Ne: 1.9e-13 Eh on a 6.3e-03 Eh
    correction. Anything larger would mean the shared subtraction is being fed differently by
    the two callers, which no molecular comparison could localize.

    Ne rather than a heavy atom: one sub-second four-component solve on each side.
    """
    from kuiva.amf.correction import amf_correction
    from kuiva.interface.pyscf_bridge import molecular_mean_field

    mol = _mol("Ne 0 0 0")
    h_mmf, w_mmf, info = molecular_mean_field(mol)
    amf = amf_correction(mol, method="x2camf")

    assert info["converged"]
    scale = float(np.max(np.abs(amf.h_sf)))
    assert scale > 1e-3, "the correction must be non-trivial or this test is vacuous"
    assert np.max(np.abs(h_mmf - amf.h_sf)) < 1e-10 * scale
    assert np.max(np.abs(w_mmf - amf.w)) < 1e-8 * float(np.max(np.abs(amf.w)))


def test_selecting_mmf_warns_that_it_is_a_benchmark_method(kuiva_caplog):
    """The user decision behind X2C-mmf: a method that is not for production must say so where it
    is chosen, not only in documentation nobody reads at the point of choosing."""
    with kuiva_caplog.at_level("WARNING"):
        ingest_spin_orbit(_mol("Ne 0 0 0"), screening="mmf")
    messages = [r.message for r in kuiva_caplog.records]
    assert any("EXPERIMENTAL BENCHMARK" in m for m in messages)
    assert any("not intended for production" in m for m in messages)


def test_mmf_ingestion_records_itself_and_keeps_the_structural_guarantees():
    """mmf must arrive through the same seam as every other screening: recorded in the
    provenance, and time-reversal even to machine precision."""
    mol = _mol("H 0 0 0; F 0 0 0.917")
    soc = ingest_spin_orbit(mol, screening="mmf")

    assert soc.method == "X2C-mmf" and soc.screening.method == "mmf"
    assert soc.screening.applied
    assert json.loads(json.dumps(soc.provenance()))["screening"]["method"] == "mmf"
    # ⚠ backend_version is a contract field, not a scratch pad for the 4c dimension.
    assert soc.screening.backend_version and "." in soc.screening.backend_version

    _, relative = time_reversal_residual(soc.hamiltonian())
    assert relative < 1e-10
    assert np.max(np.abs(soc.h_sf - soc.h_sf.T)) / np.max(np.abs(soc.h_sf)) < 1e-13


def test_mmf_and_x2camf_differ_on_a_molecule_by_the_atomic_approximation():
    """⚠ What mmf is *for*. On an atom the two agree to 1e-13; on a molecule they must not,
    and the difference is precisely the atom-diagonal approximation X2CAMF makes — the
    off-atom blocks of the two-electron picture change, which X2CAMF sets to zero.

    Measured on HF: the *peak* of `dw` agrees to 0.46 %, but the largest elementwise difference
    is 15 % of it, i.e. the atomic approximation is excellent where the operator is large
    (atom-centred, core) and worst exactly where X2CAMF has nothing.
    """
    from kuiva.amf.correction import amf_correction
    from kuiva.interface.pyscf_bridge import molecular_mean_field

    mol = _mol("H 0 0 0; F 0 0 0.917")
    _, w_mmf, _ = molecular_mean_field(mol)
    amf = amf_correction(mol, method="x2camf")

    peak_mmf, peak_amf = float(np.max(np.abs(w_mmf))), float(np.max(np.abs(amf.w)))
    assert abs(peak_mmf - peak_amf) / peak_amf < 0.02          # peaks agree to ~0.5%
    difference = float(np.max(np.abs(w_mmf - amf.w)))
    assert difference / peak_amf > 0.01                        # ...but they are not the same
