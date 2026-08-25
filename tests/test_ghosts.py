"""Ghost atoms: basis functions with no nucleus, no electrons and no mass.

A ghost is what a counterpoise correction is made of, and it is also the thing most likely to
be handled by accident rather than on purpose: it looks like an atom to every loop over atoms
and is not one to any of the physics. The tests here are mostly about the consumers that have
to *skip* it — the atomic mean field, the free-atom reference orbitals, the nuclear model, the
gauge origin — because each of those would otherwise ask the chemistry of an element called
``GHOST-Cl`` and get a plausible answer to a question nobody asked.

⚠ The single assertion that matters most is that a ghost changes **no** electron count, **no**
nuclear repulsion and **no** centre of mass, while changing the AO basis. That is the whole
definition, and a basis-set superposition error is exactly the number it makes measurable.
"""
from __future__ import annotations

import numpy as np
import pytest

import kuiva
from kuiva.amf import atomic
from kuiva.basis.atommap import parse_atom_key
from kuiva.basis.ghosts import GHOST_PREFIX, ghost_element, is_ghost, normalize_symbol
from kuiva.interface.pyscf_bridge import build_mole, gauge_origin_for

BASIS = "x2c-SVPall-2c"

#: The four-component atomic solve is a subject here: one test asserts that a ghost adds no
#: solve, which a replayed stage checkpoint would answer with a previous run's count.
pytestmark = pytest.mark.stage_under_test("amf_atomic")


# --- The vocabulary -----------------------------------------------------------------------

@pytest.mark.parametrize("written,canonical", [
    ("ghost-Cl", "ghost-Cl"), ("GHOST-cl", "ghost-Cl"), ("ghost:Cl", "ghost-Cl"),
    ("x-Cl", "ghost-Cl"), ("X-Cl2", "ghost-Cl2"), ("ghostCl", "ghost-Cl"),
])
def test_the_accepted_spellings_are_one_symbol(written, canonical):
    """Three spellings reach this program from three places (its own documentation, PySCF's,
    and other codes'). They are normalized once, so nothing downstream compares strings that
    were written differently."""
    assert normalize_symbol(written) == canonical
    assert is_ghost(written)


@pytest.mark.parametrize("symbol", ["Cl", "Xe", "X", "H", "Cl2"])
def test_a_real_atom_is_not_a_ghost(symbol):
    """⚠ ``Xe`` starts with the ghost prefix ``x`` and is xenon. A prefix test that got this
    wrong would silently delete a nucleus."""
    assert not is_ghost(symbol)


def test_the_element_is_recoverable_from_the_label():
    assert ghost_element("ghost-Cl2") == "Cl" and ghost_element("Cl") == "Cl"
    assert normalize_symbol("cl").startswith(GHOST_PREFIX) is False


# --- What a ghost is, and is not ----------------------------------------------------------

def test_a_ghost_adds_basis_functions_and_nothing_else():
    plain = build_mole(kuiva.Molecule(atoms=[("Ne", (0., 0., 0.))], basis=BASIS))
    with_ghost = build_mole(kuiva.Molecule(
        atoms=[("Ne", (0., 0., 0.)), ("ghost-Ar", (0., 0., 4.))],
        basis={"Ne": BASIS, "ghost-Ar": BASIS}))
    assert with_ghost.nao > plain.nao                      # it carries functions
    assert with_ghost.nelectron == plain.nelectron         # and no electrons
    assert with_ghost.atom_charge(1) == 0                  # and no nucleus
    assert float(np.asarray(with_ghost.atom_mass_list())[1]) == 0.0
    assert with_ghost.energy_nuc() == pytest.approx(plain.energy_nuc(), abs=1e-12)
    # ...so the gauge origin is the molecule's own, which is what lets a counterpoise pair
    # share one origin and therefore one set of property operators.
    o_plain, _ = gauge_origin_for(plain)
    o_ghost, _ = gauge_origin_for(with_ghost)
    assert np.allclose(o_plain, o_ghost, atol=1e-12)


def test_a_molecule_of_only_ghosts_has_no_centre_of_mass():
    """Refused rather than returning a NaN that every moment in a property file would then
    carry."""
    mol = build_mole(kuiva.Molecule(atoms=[("ghost-Ne", (0., 0., 0.))],
                                    basis={"ghost-Ne": BASIS}))
    with pytest.raises(ValueError, match="no mass"):
        gauge_origin_for(mol)
    with pytest.raises(ValueError, match="no nuclear charge"):
        gauge_origin_for(mol, "charge")


def test_a_ghost_is_addressed_by_its_own_label_and_not_by_its_element():
    """⚠ ``basis={"Cl": ...}`` must not reach a ghost chlorine. A ghost and a real atom of one
    element are different things to every consumer downstream, so a key covering both would be
    a way to state one basis and get two."""
    symbols = ["ghost-Cl", "Ne"]
    assert parse_atom_key("ghost-Cl", symbols) == ("element", "ghost-Cl")
    assert parse_atom_key("ghost-Cl1", symbols) == ("atom", 0)
    with pytest.raises(ValueError, match="names no atom"):
        parse_atom_key("Cl", symbols)


def test_two_ghosts_of_one_element_can_carry_different_bases():
    """They are decorated exactly as two real atoms of one element are, which is what makes a
    per-atom basis statement possible for them at all."""
    mol = build_mole(kuiva.Molecule(
        atoms=[("ghost-Ne", (0., 0., 0.)), ("ghost-Ne", (0., 0., 4.)), ("He", (0., 0., 8.))],
        basis={"ghost-Ne1": "x2c-SVPall-2c", "ghost-Ne2": "x2c-TZVPall-2c", "He": BASIS}))
    labels = mol.__dict__["_kuiva_atom_labels"]
    assert labels[0] != labels[1] and labels[0].startswith(GHOST_PREFIX)
    slices = mol.aoslice_by_atom()
    assert slices[0][3] - slices[0][2] != slices[1][3] - slices[1][2]


# --- What a ghost must not reach ----------------------------------------------------------

def test_the_atomic_mean_field_skips_ghosts():
    """⚠ A ghost has no nucleus, so it has no two-electron picture change — and its own
    ``atom_pure_symbol`` is the ghost label, not an element, so a consumer that did not skip it
    would ask the registry about an element called GHOST-Ar."""
    from kuiva.amf.correction import amf_correction

    mol = build_mole(kuiva.Molecule(
        atoms=[("Ne", (0., 0., 0.)), ("ghost-Ar", (0., 0., 4.))],
        basis={"Ne": BASIS, "ghost-Ar": BASIS}))
    atomic.clear_cache()
    correction = amf_correction(mol)
    assert correction.elements == ("Ne",)
    assert atomic.cache_statistics()["solves"] == 1        # one per *real* unique element
    p0, p1 = mol.aoslice_by_atom()[1][2:4]
    assert np.max(np.abs(correction.h_sf[p0:p1, p0:p1])) == 0.0
    assert np.max(np.abs(correction.w[:, p0:p1, p0:p1])) == 0.0
    atomic.clear_cache()


def test_a_ghost_does_not_make_a_molecule_look_mixed_in_its_nuclear_model():
    """The nuclear model is a property of nuclei. A ghost has none, so it must not be able to
    trip the mixed-model refusal — which it would if it were merely read off the atom table."""
    for model in ("point", "gaussian"):
        mol = build_mole(kuiva.Molecule(
            atoms=[("Ne", (0., 0., 0.)), ("ghost-Ar", (0., 0., 4.))],
            basis={"Ne": BASIS, "ghost-Ar": BASIS}, nuclear_model=model))
        assert atomic.nuclear_model_of(mol) == model


# --- What it is for -----------------------------------------------------------------------

def test_a_counterpoise_pair_measures_a_basis_set_superposition_error():
    """The point of the feature, as one number.

    The same neon, the same nuclei, the same electrons — in the dimer's basis instead of its
    own. The difference is the energy neon gains from functions that carry no electrons of
    their own: negative by construction (a variational calculation given more functions), and
    small for a well-separated pair in a decent basis.
    """
    plain = kuiva.ScalarSCF(kuiva.Molecule(atoms=[("Ne", (0., 0., 0.))], basis=BASIS),
                            memory_gb=4.0, screening="none").run()
    ghosted = kuiva.ScalarSCF(kuiva.Molecule(
        atoms=[("Ne", (0., 0., 0.)), ("ghost-Ar", (0., 0., 4.))],
        basis={"Ne": BASIS, "ghost-Ar": BASIS}), memory_gb=4.0, screening="none").run()
    bsse = ghosted.energy - plain.energy
    assert ghosted.data.nelec == plain.data.nelec
    assert -1e-2 < bsse < 0.0, "BSSE must be negative and small; got {:.3e} Eh".format(bsse)


def test_the_charge_partition_reports_the_density_that_leaked_onto_a_ghost():
    """A ghost has no free-atom reference, and that is what makes its population meaningful:
    it is density borrowed from a basis carrying no electrons, i.e. the superposition error
    resolved per centre. Its nuclear charge is zero, so its reported charge is minus that."""
    from kuiva.props.population import atomic_reference_charges

    run = kuiva.ScalarSCF(kuiva.Molecule(
        atoms=[("Ne", (0., 0., 0.)), ("ghost-Ar", (0., 0., 4.))],
        basis={"Ne": BASIS, "ghost-Ar": BASIS}),
        memory_gb=4.0, screening="none", atomic_reference=True).run()
    reference = run.data.atomic_reference
    assert sorted(reference.entries) == ["Ne"]
    q = atomic_reference_charges(run.data.mo_coeff, run.data.s_ao, run.data.ao_layout,
                                 reference=reference, occupation=run.data.mo_occ)
    assert q.population.sum() == pytest.approx(sum(run.data.nelec), abs=1e-8)
    assert q.charge[1] < 0.0                     # the ghost holds a little density
    assert q.charge[0] == pytest.approx(-q.charge[1], abs=1e-10)
