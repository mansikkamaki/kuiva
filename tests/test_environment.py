"""Point-charge embedding: what the environment is allowed to change, and what it must not.

A single-molecule magnet is measured in a crystal, and a bare gas-phase ion is a different
system with a different ligand field. This is the smallest honest way to say so — and the
smallest is the point: an embedding of this kind adds **one** term to the one-electron
Hamiltonian and **one** classical constant, and nothing else in the program may learn that the
charges exist. The multireference layer is handed a one-electron Hamiltonian, as it always was.

The tests are in four groups: the arithmetic (against an independent implementation of the
same embedding), the boundaries (what it must not move — the gauge origin, the nuclear
repulsion, the electron count), the statements (units, provenance, refusals), and the one
place where an embedded calculation is genuinely *not* the vacuum one it looks like — its
symmetry.
"""
from __future__ import annotations

import numpy as np
import pytest

import kuiva
from kuiva.interface.environment import (Environment, MIN_NUCLEUS_DISTANCE,
                                         NEGLIGIBLE_CHARGE, PointCharges,
                                         embedding_operator)
from kuiva.interface.pyscf_bridge import build_mole, gauge_origin_for, ingest_spin_orbit

BASIS = "x2c-SVPall-2c"
FIELD = [(0.7, (0., 0., 5.0)), (-0.3, (0., 2.0, -4.0))]


def neon(**kw):
    return kuiva.Molecule(atoms=[("Ne", (0., 0., 0.))], basis=BASIS, unit="Bohr", **kw)


# --- The arithmetic ------------------------------------------------------------------------

def test_the_embedded_energy_matches_an_independent_implementation():
    """⚠ The decisive test, and it is decisive because the two sides share no code: PySCF's
    own QM/MM embedding of the same charges on the same scalar-X2C SCF. An in-house check of
    an in-house potential could agree to any number of digits and still be one wrong sign in
    two places."""
    from pyscf import gto, qmmm, scf

    run = kuiva.ScalarSCF(neon(environment=Environment(point_charges=FIELD)),
                          memory_gb=4.0, screening="none").run()
    charges = np.array([q for q, _ in FIELD])
    coords = np.array([r for _, r in FIELD])
    mol = gto.M(atom="Ne 0 0 0", basis=run.data.molecule.basis, unit="Bohr", verbose=0)
    mf = qmmm.mm_charge(scf.RHF(mol).sfx2c1e(), coords, charges, unit="Bohr")
    mf.conv_tol = 1e-10
    reference = mf.kernel()
    assert run.energy == pytest.approx(reference, abs=1e-10)


def test_the_charge_nucleus_energy_is_the_analytic_one():
    """A sum of q Z / d, which is worth asserting because it is the term most easily left out:
    the potential alone gives a total that is wrong by a constant nobody would notice."""
    mol = build_mole(neon())
    op = embedding_operator(mol, Environment(point_charges=FIELD, unit="bohr").resolved())
    expected = 10.0 * (0.7 / 5.0 - 0.3 / np.sqrt(2.0 ** 2 + 4.0 ** 2))
    assert op.e_nuclear == pytest.approx(expected, rel=1e-12)


def test_the_potential_reaches_the_two_component_hamiltonian_exactly_once():
    """The embedding term is added in the AO basis, beside the two-electron picture change and
    for the same reason. ⚠ Its size in ``h_sf`` must be *exactly* the operator — not twice it,
    which is what a term added in two places looks like, and which no invariant would catch."""
    mol = build_mole(neon())
    op = embedding_operator(mol, Environment(point_charges=FIELD, unit="bohr").resolved())
    vacuum = ingest_spin_orbit(mol, screening="none")
    embedded = ingest_spin_orbit(mol, screening="none", embedding=op)
    assert np.allclose(embedded.h_sf - vacuum.h_sf, op.h_sf, atol=1e-14)
    # The bare operator is spin-free, so the spin-orbit part is bitwise untouched.
    assert np.array_equal(embedded.w, vacuum.w)


def test_a_neutral_symmetric_field_leaves_the_nuclei_alone():
    """A charge-nucleus energy of zero is not a trivial case: it is the one where a missing
    term would be invisible in the total and visible only in the density."""
    field = [(0.5, (0., 0., 5.0)), (-0.5, (0., 0., -5.0))]
    run = kuiva.ScalarSCF(neon(environment=Environment(point_charges=field)),
                          memory_gb=4.0, screening="none").run()
    assert run.data.e_embedding == pytest.approx(0.0, abs=1e-12)
    assert run.energy != pytest.approx(
        kuiva.ScalarSCF(neon(), memory_gb=4.0, screening="none").run().energy, abs=1e-8)


# --- The boundaries ------------------------------------------------------------------------

def test_vacuum_is_bitwise_what_it_was():
    """No environment must mean no change of any kind — the feature is opt-in and every
    committed number in this project was produced without it."""
    plain = build_mole(kuiva.Molecule(atoms=[("Ne", (0., 0., 0.))], basis=BASIS))
    assert embedding_operator(plain, None) is None
    assert embedding_operator(plain, Environment()) is None
    soc = ingest_spin_orbit(plain, screening="none")
    assert not soc.embedding.embedded
    assert soc.provenance()["embedding"]["embedded"] is False


def test_the_charges_move_neither_the_gauge_origin_nor_the_nuclear_repulsion():
    """⚠ Deliberate: the orbital angular momentum in a property file is defined about the
    *molecule*, so an embedded calculation and a gas-phase one of the same complex stay
    comparable. And the nuclear repulsion stays the nuclei's own, with the charge-nucleus term
    reported on its own line."""
    plain, embedded = build_mole(neon()), build_mole(neon(
        environment=Environment(point_charges=FIELD)))
    assert np.allclose(gauge_origin_for(plain)[0], gauge_origin_for(embedded)[0], atol=1e-14)
    run = kuiva.ScalarSCF(neon(environment=Environment(point_charges=FIELD)),
                          memory_gb=4.0, screening="none").run()
    assert run.data.e_nuc == pytest.approx(0.0, abs=1e-12)      # one atom: no repulsion
    assert run.data.e_embedding != 0.0                          # ...and the field is not in it


def test_the_electron_count_is_untouched():
    run = kuiva.ScalarSCF(neon(environment=Environment(point_charges=FIELD)),
                          memory_gb=4.0, screening="none").run()
    assert run.data.nelec == (5, 5)


# --- The statements ------------------------------------------------------------------------

def test_the_unit_is_the_molecules_own_unless_the_field_states_one():
    """⚠ The trap this design exists to close: a field copied out of a crystallographic file
    is in Angstrom and a geometry may be in either. The same numbers in the two units are
    different physics, and inheriting the molecule's is the only inference that cannot be
    wrong."""
    field = [(1.0, (0., 0., 5.0))]
    inherited = build_mole(kuiva.Molecule(
        atoms=[("Ne", (0., 0., 0.))], basis=BASIS, unit="Angstrom",
        environment=Environment(point_charges=field)))
    env = kuiva.Molecule(atoms=[("Ne", (0., 0., 0.))], basis=BASIS, unit="Angstrom",
                         environment=Environment(point_charges=field)).environment
    in_bohr = env.resolved("Angstrom").point_charges.coords
    assert in_bohr[0, 2] == pytest.approx(5.0 / 0.52917721092, rel=1e-9)
    stated = Environment(point_charges=field, unit="bohr").resolved("Angstrom")
    assert stated.point_charges.coords[0, 2] == pytest.approx(5.0, rel=1e-12)
    assert inherited.natm == 1


def test_a_charge_on_top_of_a_nucleus_is_refused():
    """The density polarizes onto it without bound and the SCF converges to a number that
    means nothing, so this is a refusal rather than a warning."""
    mol = build_mole(neon())
    close = Environment(point_charges=[(1.0, (0., 0., 0.1))], unit="bohr").resolved()
    with pytest.raises(ValueError, match="from a nucleus"):
        embedding_operator(mol, close)
    ok = Environment(point_charges=[(1.0, (0., 0., MIN_NUCLEUS_DISTANCE + 0.1))],
                     unit="bohr").resolved()
    assert embedding_operator(mol, ok) is not None


def test_numerically_zero_charges_are_dropped():
    """A lattice sum or an Ewald fit produces a tail of them, and an integral each is pure
    cost."""
    mol = build_mole(neon())
    env = Environment(point_charges=[(1.0, (0., 0., 5.0)),
                                     (NEGLIGIBLE_CHARGE / 10.0, (0., 0., 7.0))],
                      unit="bohr").resolved()
    assert embedding_operator(mol, env).record.n_charges == 1


def test_the_provenance_identifies_the_field_without_storing_it():
    """A lattice does not belong in a file header; a digest identifies it exactly, and two
    files carrying the same one were embedded in the same field."""
    mol = build_mole(neon())
    env = Environment(point_charges=FIELD, unit="bohr", label="test field").resolved()
    op = embedding_operator(mol, env)
    soc = ingest_spin_orbit(mol, screening="none", embedding=op)
    record = soc.provenance()["embedding"]
    assert record["embedded"] and record["n_charges"] == 2
    assert record["net_charge"] == pytest.approx(0.4)
    assert record["digest"] == env.point_charges.digest()
    assert record["label"] == "test field"
    assert soc.transform(np.eye(soc.nao)).embedding == soc.embedding
    other = Environment(point_charges=[(0.7, (0., 0., 5.1)), (-0.3, (0., 2.0, -4.0))],
                        unit="bohr").resolved()
    assert other.point_charges.digest() != env.point_charges.digest()


def test_a_malformed_field_is_refused_at_construction():
    with pytest.raises(ValueError, match=r"\(q, \(x, y, z\)\)"):
        Environment(point_charges=[(1.0, 2.0)])
    with pytest.raises(ValueError, match="one of each"):
        PointCharges(charges=[1.0, 2.0], coords=[[0.0, 0.0, 1.0]])


# --- The one thing an embedding really does change ------------------------------------------

def test_a_field_that_breaks_a_symmetry_removes_it_from_the_labelling():
    """⚠ The molecule is not the whole system. A single charge on the z axis leaves C2(z) and
    destroys inversion; a reference labelled from the nuclei alone would carry irreps its
    states do not have."""
    run = kuiva.ScalarSCF(
        neon(point_group="auto", environment=Environment(point_charges=[(0.4, (0., 0., 5.0))])),
        memory_gb=4.0, screening="none").run()
    assert run.data.symmetry.detected == ("C2(z)",)
    assert run.data.symmetry.full_group is None          # classification declines to guess


def test_a_field_that_keeps_the_symmetry_keeps_the_labels():
    run = kuiva.ScalarSCF(
        neon(point_group="auto",
             environment=Environment(point_charges=[(0.4, (0., 0., 5.0)),
                                                    (0.4, (0., 0., -5.0))])),
        memory_gb=4.0, screening="none").run()
    assert set(run.data.symmetry.detected) == {"C2(z)", "i", "sigma(xy)"}


# --- The picture change (off by default, measured) -------------------------------------------

def test_the_bare_operator_is_the_default_and_is_spin_free():
    mol = build_mole(neon())
    op = embedding_operator(mol, Environment(point_charges=FIELD, unit="bohr").resolved())
    assert op.w is None and not op.record.picture_change


def test_the_picture_changed_operator_differs_and_carries_spin_orbit_coupling():
    """The alternative exists so the size of the approximation can be measured rather than
    asserted. ⚠ What it must *not* be is large: the charges sit outside the molecule, so their
    potential is smooth exactly where the small component is."""
    mol = build_mole(neon())
    bare = embedding_operator(mol, Environment(point_charges=FIELD, unit="bohr").resolved())
    pc = embedding_operator(mol, Environment(point_charges=FIELD, unit="bohr",
                                             picture_change=True).resolved())
    assert pc.record.picture_change and pc.w is not None
    rel = np.max(np.abs(bare.h_sf - pc.h_sf)) / np.max(np.abs(bare.h_sf))
    assert rel < 1e-3, "the embedding picture change is {:.2e} relative".format(rel)
    assert np.max(np.abs(pc.w)) > 0.0
