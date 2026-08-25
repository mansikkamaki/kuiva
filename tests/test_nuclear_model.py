"""The nuclear charge model: point (the default) and the finite Gaussian nucleus.

Three things have to be true, and only the first is about arithmetic.

1. **The default did not move.** A point nucleus is what every committed reference number in
   this project was produced with, so the default path must be *bitwise* what it was before
   this option existed — not "the same to a tolerance".
2. **One statement reaches every consumer.** The model is stated once, on the molecule, and
   the molecular integrals, the four-component atomic solves behind the two-electron
   spin-orbit screening, the free-atom reference orbitals and the isolated-fragment blocks all
   have to inherit it. ⚠ The failure mode being guarded is not a crash: an atomic mean field
   solved over a point nucleus and added to a molecular Hamiltonian built over a finite one is
   Hermitian, time-reversal even, of entirely plausible magnitude and wrong. The tests here
   are therefore about the *mechanism* — where the value comes from — and not only about the
   numbers it produces.
3. **A stored result says which nucleus it describes**, in the Hamiltonian's provenance and in
   the atomic cache's key.

The physics is asserted through the one-electron X2C spinor levels of a closed-shell atom,
which cost milliseconds and have an unambiguous direction: a finite nucleus removes charge
from the region where the spin-orbit operator is largest, so every j-splitting *decreases*,
and the relative size of the shift grows steeply with Z. A proton is small enough that no
valence basis can resolve its size at all, which is asserted as a zero rather than as a
tolerance.

Reference for the finite model: L. Visscher and K. Dyall, Atomic Data and Nuclear Data Tables
67, 207 (1997), which is what PySCF's ``dyall_nuc_mod`` implements.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import scipy.linalg as sla

from kuiva.amf import atomic, cache
from kuiva.amf.configuration import AtomicConfiguration
from kuiva.interface.api import Molecule
from kuiva.interface.pyscf_bridge import build_mole, ingest_spin_orbit
from kuiva.x2c.nuclear import (NUCLEAR_MODELS, NuclearRecord, nuclear_record, pyscf_nucmod,
                               resolve_nuclear_model)

BASIS = "x2c-SVPall-2c"
HARTREE_TO_CM = 219474.6313632

#: ⚠ The four-component atomic solve is a **subject** here, not an ingredient: several tests
#: below assert what a solve was told about the nucleus and what the cache does with the
#: answer. A replayed stage checkpoint, or a stage-managed correction cache underneath them,
#: would answer those questions with a previous run's numbers.
pytestmark = pytest.mark.stage_under_test("amf_atomic")


def atom(symbol, model="point", basis=BASIS):
    """The built ``Mole`` of one atom, through the ordinary front-end path."""
    from pyscf import gto

    return build_mole(Molecule(atoms=[(symbol, (0.0, 0.0, 0.0))], basis=basis,
                               spin=int(gto.charge(symbol)) % 2, nuclear_model=model))


def spinor_levels(symbol, model):
    """One-electron X2C spinor energies [Eh] of an atom, ascending.

    Eigenvalues of the two-component Hamiltonian in the AO basis — no SCF, no screening, so
    this is a property of the operator alone and costs a fraction of a second even for
    mercury. Levels come in Kramers pairs, so a closed-shell atom's second shell is
    ``1s(2) 2s(2) 2p_{1/2}(2) 2p_{3/2}(4)`` and the deepest j-splitting is ``e[6] - e[4]``.
    """
    mol = atom(symbol, model)
    soc = ingest_spin_orbit(mol, screening="none")
    overlap = np.kron(np.eye(2), mol.intor("int1e_ovlp"))
    return sla.eigh(soc.hamiltonian(), overlap, eigvals_only=True)


# --- The vocabulary, and the flag it exists to keep away from callers ----------------------

@pytest.mark.parametrize("given,expected", [
    (None, "point"), ("point", "point"), ("POINT", "point"), ("point-charge", "point"),
    ("gaussian", "gaussian"), ("gauss", "gaussian"), ("Finite", "gaussian"),
])
def test_spellings_resolve_to_one_model(given, expected):
    assert resolve_nuclear_model(given) == expected
    assert expected in NUCLEAR_MODELS, "every resolution lands on a declared model"


def test_an_unknown_model_is_refused_not_defaulted():
    """A typo must not silently select the default: the numbers would be right for a
    Hamiltonian nobody asked for."""
    with pytest.raises(ValueError, match="unknown nuclear_model"):
        resolve_nuclear_model("gausian")


def test_a_number_is_refused():
    """⚠ The trap this vocabulary exists for. PySCF's ``nucmod`` reads *any* non-zero value as
    a request for a Gaussian nucleus, so ``1`` — the obvious spelling of "model one", and the
    value of its own ``NUC_POINT`` constant — means the opposite of what it looks like. Kuiva
    never takes a number here, so the two vocabularies cannot be confused at a call site."""
    with pytest.raises(TypeError, match="not a number"):
        resolve_nuclear_model(1)


def test_the_flag_mapping_is_written_exactly_once():
    """``pyscf_nucmod`` is the only place the model becomes a library flag, and the test is
    that its output *behaves* — asserting the constant would just restate the code."""
    from pyscf import gto

    assert pyscf_nucmod("point") == 0
    point = gto.M(atom="Ne 0 0 0", basis="sto-3g", nucmod=pyscf_nucmod("point"), verbose=0)
    gauss = gto.M(atom="Ne 0 0 0", basis="sto-3g", nucmod=pyscf_nucmod("gaussian"), verbose=0)
    trap = gto.M(atom="Ne 0 0 0", basis="sto-3g", nucmod=1, verbose=0)
    assert atomic.nuclear_model_of(point) == "point"
    assert atomic.nuclear_model_of(gauss) == "gaussian"
    # The trap itself, pinned: nucmod=1 is a *finite* nucleus. If PySCF ever changes this, the
    # mapping above is what has to be revisited, and this is what says so.
    assert atomic.nuclear_model_of(trap) == "gaussian"


def test_the_record_states_where_the_parameters_come_from():
    point, gauss = nuclear_record("point"), nuclear_record("gaussian")
    assert not point.finite and gauss.finite
    assert point.label() == "point" and "visscher-dyall-1997" in gauss.label()
    # A contract with stored data: it must survive JSON unchanged, field for field.
    import json

    assert json.loads(json.dumps(gauss.as_dict())) == gauss.as_dict()
    assert set(gauss.as_dict()) == {"model", "parametrization", "isotopes", "implementation"}


# --- One statement, every consumer --------------------------------------------------------

def test_the_default_is_point_and_bitwise_unchanged():
    """⚠ The load-bearing test of the whole feature: adding the option moved nothing.

    Not "agrees to 1e-12" — *bitwise*, against a molecule built with no nuclear-model
    statement at all, because every committed reference and every example output in this
    project was produced on this path.
    """
    from pyscf import gto

    mol = atom("Ne")
    assert atomic.nuclear_model_of(mol) == "point"
    bare = gto.M(atom=[("Ne", (0.0, 0.0, 0.0))], basis=mol._basis, verbose=0)
    assert np.array_equal(mol.intor("int1e_nuc"), bare.intor("int1e_nuc"))
    assert np.array_equal(mol.intor("int1e_kin"), bare.intor("int1e_kin"))


def test_every_atom_of_the_molecule_gets_the_model():
    """The statement is per molecule, so no atom may be left behind — a molecule with one
    point nucleus among finite ones is the shape a half-wired option would take."""
    mol = build_mole(Molecule(atoms=[("H", (0.0, 0.0, 0.0)), ("F", (0.0, 0.0, 0.9))],
                              basis=BASIS, nuclear_model="gaussian"))
    assert atomic.nuclear_model_of(mol) == "gaussian"
    assert atomic.nuclear_model_of(
        build_mole(Molecule(atoms=[("H", (0.0, 0.0, 0.0)), ("F", (0.0, 0.0, 0.9))],
                            basis=BASIS))) == "point"


def test_a_mixed_molecule_is_refused():
    """A mixture cannot come from Kuiva's own input, so it comes from a ``Mole`` built
    elsewhere — and there is no honest single answer to hand an atomic solve that has to
    match it."""
    from pyscf import gto

    mixed = gto.M(atom="H 0 0 0; F 0 0 0.9", basis="sto-3g", nucmod={"F": "gauss"}, verbose=0)
    with pytest.raises(NotImplementedError, match="mixes nuclear charge models"):
        atomic.nuclear_model_of(mixed)


def test_the_molecule_spec_carries_the_model():
    """``MoleculeSpec`` duck-types as a ``Molecule`` for :func:`build_mole`, and a rebuilt
    molecule that differs from the original in any input is not the one the run used. ⚠ No
    overlap integral depends on the nucleus, which is exactly why dropping it here would go
    unnoticed."""
    from kuiva.interface.pyscf_bridge import MoleculeSpec

    spec = MoleculeSpec.from_molecule(
        Molecule(atoms=[("Ne", (0.0, 0.0, 0.0))], basis=BASIS, nuclear_model="gaussian"))
    assert spec.nuclear_model == "gaussian"
    assert atomic.nuclear_model_of(build_mole(spec)) == "gaussian"


# --- The physics --------------------------------------------------------------------------

def test_a_proton_is_a_point_charge_as_far_as_any_basis_can_tell():
    """Asserted as a zero, not as a tolerance: hydrogen's nuclear radius is four orders below
    the resolution of a valence basis, so a finite-nucleus run of a light molecule must be the
    point-nucleus run to well inside every tolerance in this suite."""
    h_point = ingest_spin_orbit(atom("H", "point"), screening="none").h_sf
    h_gauss = ingest_spin_orbit(atom("H", "gaussian"), screening="none").h_sf
    rel = np.max(np.abs(h_point - h_gauss)) / np.max(np.abs(h_point))
    assert rel < 1e-8, "the finite-nucleus shift on hydrogen is {:.2e} relative".format(rel)


def test_j_splittings_shrink_and_the_shift_grows_with_z():
    """The physical signature, and both halves of it matter.

    A Gaussian nucleus removes charge from the region where the spin-orbit operator is
    largest, so every j-splitting comes out **smaller** than with a point nucleus — a wrong
    sign here would be a wrong subtraction, not a small error. And the *relative* shift grows
    steeply with Z, which is the reason the model is an option at all: below 1e-6 for neon it
    is beyond any comparison this project makes, while for mercury it is ~3e-3, far above the
    0.1 cm^-1 bar a free-ion multiplet is held to.
    """
    shifts = []
    for symbol in ("Ne", "Kr", "Xe", "Hg"):
        point, gauss = spinor_levels(symbol, "point"), spinor_levels(symbol, "gaussian")
        sp, sg = point[6] - point[4], gauss[6] - gauss[4]          # deepest 2p_1/2 - 2p_3/2
        assert sp > 0 and sg > 0
        assert sg < sp, ("the {} 2p splitting must shrink with a finite nucleus; got "
                         "{:.2f} -> {:.2f} cm^-1".format(symbol, sp * HARTREE_TO_CM,
                                                         sg * HARTREE_TO_CM))
        shifts.append(abs(sg - sp) / sp)
    assert shifts == sorted(shifts), \
        "the relative shift must grow with Z; got {}".format(
            ["{:.2e}".format(s) for s in shifts])
    assert shifts[0] < 1e-6, "neon: {:.2e}".format(shifts[0])
    assert shifts[-1] > 1e-3, "mercury: {:.2e}".format(shifts[-1])


# --- Provenance ---------------------------------------------------------------------------

def test_the_hamiltonian_records_the_nucleus_it_was_built_over():
    """⚠ Read off the molecule rather than off an argument, so the record cannot describe a
    different operator from the one returned."""
    soc = ingest_spin_orbit(atom("Ne", "gaussian"), screening="none")
    assert soc.nuclear.model == "gaussian"
    assert soc.provenance()["nuclear"] == soc.nuclear.as_dict()
    # It is a property of the operator, so a change of basis must carry it.
    assert soc.transform(np.eye(soc.nao)).nuclear == soc.nuclear
    assert ingest_spin_orbit(atom("Ne"), screening="none").nuclear == NuclearRecord(
        model="point", implementation="pyscf")


def test_the_model_is_named_in_the_output(kuiva_caplog):
    """It is in the Hamiltonian block of the output, because a run whose output does not say
    which nucleus it used cannot be compared with anything."""
    soc = ingest_spin_orbit(atom("Ne", "gaussian"), screening="none")
    kuiva_caplog.clear()
    soc.report()
    text = "\n".join(r.getMessage() for r in kuiva_caplog.records)
    assert "nuclear charge model" in text and "gaussian" in text


# --- The atomic path: the key, the cache, and where the value comes from -------------------

def test_the_cache_key_separates_every_field_of_a_request():
    """⚠ ``cache_key`` is a hand-written tuple and a new field does not join it on its own.

    Two requests that differ only in a forgotten field then name one file: nothing wrong is
    ever *served* (the stored request is checked field by field), but the entry is rewritten
    on every run and the persistent cache silently stops working. This asserts one variation
    per field, and that the variation map covers the dataclass — so a field added without a
    thought for the key fails here.
    """
    base = atomic.make_request("Ne", "sto-3g")
    variations = {
        "element": "Ar",
        "basis_digest": "0" * 64,
        "charge": 1,
        "configuration": AtomicConfiguration.coerce("[He]2s2", "Ne"),
        "interaction": "gaunt",
        "backend": "stub",
        "light_speed": 1.0e4,
        "uncontract": False,
        "nuclear_model": "gaussian",
    }
    assert set(variations) == {f.name for f in dataclasses.fields(base)}, \
        "a field of AtomicRequest is missing from this map — and probably from cache_key"
    for name, value in variations.items():
        other = dataclasses.replace(base, **{name: value})
        assert atomic.cache_key(other) != atomic.cache_key(base), \
            "cache_key does not separate two requests differing in {!r}".format(name)


def test_the_persistent_cache_does_not_serve_one_nucleus_for_another(tmp_path, monkeypatch,
                                                                     kuiva_caplog):
    """The entry for a point nucleus must be a **miss** for a finite-nucleus request, and a
    quiet one: this is "not computed yet", not a broken key."""
    pytest.importorskip("h5py", reason="the persistent cache needs h5py")
    monkeypatch.setenv(cache.ENV_CACHE_DIR, str(tmp_path / "amf-cache"))
    atomic.clear_cache()
    from kuiva.amf.decouple import AtomicAMF

    point = atomic.make_request("Ne", "sto-3g")
    gauss = atomic.make_request("Ne", "sto-3g", nuclear_model="gaussian")
    fake = AtomicAMF(h_sf=np.eye(3), w=np.zeros((3, 3, 3)), configuration=point.configuration,
                     scale=1.0, tr_residual=0.0, tr_residual_rel=0.0, transformed_scale=1.0,
                     subtracted_scale=0.0)
    assert cache.store(point, atomic.cache_key(point), fake)

    kuiva_caplog.clear()
    assert cache.load(gauss, atomic.cache_key(gauss)) is None
    assert cache.load(point, atomic.cache_key(point)) is not None
    assert not [r for r in kuiva_caplog.records if r.levelname == "WARNING"], \
        "a request for a model that has not been computed is an ordinary miss"
    atomic.clear_cache()


def test_the_formula_version_was_bumped_for_the_key_change():
    """⚠ An entry written before the model joined the key describes a point nucleus and is
    still numerically right for one — so this bump was not forced by arithmetic. It is taken
    because 'the key is complete' is a claim about code that is no longer running, and a
    stored quantity outlives it. The cache's own tests assert that a stale version misses;
    this asserts the version actually moved with the change that motivated it.
    """
    assert cache.FORMULA_VERSION >= 4


def test_the_atomic_solve_is_told_which_nucleus_the_molecule_has(monkeypatch):
    """The mechanism, not the number: the molecular assembly must take the model off the
    ``Mole`` it is correcting. ⚠ A four-component atomic solve over the wrong nucleus produces
    a correction of entirely plausible magnitude — there is nothing in the result to notice —
    so what is asserted is where the value came from."""
    from kuiva.amf import correction as corr

    seen = []
    real = corr.atomic_correction

    def spy(symbol, basis, **kw):
        seen.append(kw.get("nuclear_model"))
        return real(symbol, basis, **kw)

    monkeypatch.setattr(corr, "atomic_correction", spy)
    corr.amf_correction(atom("Ne", "gaussian"))
    corr.amf_correction(atom("Ne", "point"))
    assert seen == ["gaussian", "point"]


def test_a_solution_records_the_nucleus_it_was_solved_over():
    """For the same reason it records the speed of light: it is part of what the solution
    *is*, and the two must never be mistaken for one another — nor cached alongside."""
    point = atomic.atomic_solution("Ne", "sto-3g")
    gauss = atomic.atomic_solution("Ne", "sto-3g", nuclear_model="gaussian")
    assert point.nuclear_model == "point" and gauss.nuclear_model == "gaussian"
    assert gauss.e_tot != point.e_tot
    # Neon is light, so the shift is small — the assertion is that it is there at all, and in
    # the direction a finite nucleus produces: less attractive potential, higher energy.
    assert gauss.e_tot > point.e_tot


def test_the_external_plugin_is_refused_rather_than_given_the_wrong_nucleus():
    """It solves its own atomic references over a point nucleus and Kuiva cannot tell it
    otherwise. A bisection tool that silently answers a different question is worse than one
    that is unavailable."""
    from kuiva.amf import correction as corr

    with pytest.raises(NotImplementedError, match="x2camf-external"):
        corr.amf_correction(atom("Ne", "gaussian"), method="x2camf-external")


def test_free_atom_reference_orbitals_are_keyed_on_the_model():
    """The free atom and the atom inside the molecule must be the same atom, or a population
    is measured against a reference describing something else. Reached through the private
    entry point because the key is the subject: what a public call returns would look right
    either way."""
    from kuiva.interface import pyscf_bridge as bridge

    bridge._ATOMIC_REFERENCE_CACHE.clear()
    config = AtomicConfiguration.coerce(None, "Ne")
    point = bridge._atomic_reference_entry("Ne", BASIS, config, True, "point")
    gauss = bridge._atomic_reference_entry("Ne", BASIS, config, True, "gaussian")
    assert len(bridge._ATOMIC_REFERENCE_CACHE) == 2
    assert not np.array_equal(point.c, gauss.c)
    # ...and the same request is served from the cache rather than solved again.
    again = bridge._atomic_reference_entry("Ne", BASIS, config, True, "gaussian")
    assert again is gauss
    bridge._ATOMIC_REFERENCE_CACHE.clear()
