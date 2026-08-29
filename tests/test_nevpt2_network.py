"""Parity of the network-backed SC-NEVPT2 provider against the conventional-CI one.

The seam was designed for exactly this check: the classes read only registered
primitives, so the two providers are compared primitive by primitive on a reference both
represent exactly (a saturating bond dimension), and then class by class through the one
shared driver loop. The two implementations share the density *algebra* (one set of pure
functions) but nothing of the reference machinery — determinant vectors and shifted
spaces on one side, applied Jordan–Wigner strings and tree contractions on the other —
which is what makes agreement a check rather than a tautology.

⚠ What the network E2 is: **PARTIAL, six of eight classes** — the primed single-external
classes are not served (the recorded scope of ``kuiva/pt/network.py``), the driver skips
them with a warning, and the result says so. The tests pin that the partiality is loud,
not that it is absent.
"""
import numpy as np
import pytest

from kuiva.ci.strings import CASSpace
from kuiva.dmrg.graph import NetworkGraph
from kuiva.dmrg.solver import DMRGSolver
from kuiva.dmrg.sweep import random_state, solve_ttn
from kuiva.dmrg.ttno import TTNOTemplate, fermion_term
from kuiva.mcscf.casci import FullCISolver, casci
from kuiva.mcscf.orbopt import CASIntegrals
from kuiva.pt.contractions import CIContractionProvider
from kuiva.pt.network import NetworkContractionProvider, sc_nevpt2_dmrg
from kuiva.pt.nevpt2 import sc_nevpt2

from test_ci_strings import random_spinor_integrals

#: The classes the network provider serves, and the two it deliberately does not.
SERVED = ("Sijrs", "Sijr", "Srsi", "Srs", "Sij", "Sir")
MISSING = ("Sr", "Si")

TOL = 1e-8


# --- provider-level parity on one exactly represented state ---------------------------------

@pytest.fixture(scope="module")
def providers():
    """The two providers over the SAME state: CI vector and saturated network."""
    n, k = 6, 2
    h, eri = random_spinor_integrals(n, seed=81)
    tpl = TTNOTemplate(NetworkGraph.path(n))
    op = tpl.fill(h, eri)
    state = random_state(op, k, 200, n_roots=1, rng=np.random.default_rng(81))
    result = solve_ttn(op, state, max_bond=200, boundary_check=0, report=False,
                       conv_tol=1e-12)
    assert result.converged
    ci_solver = FullCISolver(n, k, n_states=1, enforce_kramers=False)
    ref = ci_solver.solve_active(h, eri)
    assert abs(result.energies[0] - ref.energies[0]) < TOL
    ci = CIContractionProvider(CASSpace(n, k), ref.vectors[0], h, eri)
    from kuiva.dmrg.density import state_rdms
    gamma, gamma2 = state_rdms(tpl, state)[0]
    net = NetworkContractionProvider(tpl, state, 0, h, eri, gamma=gamma, gamma2=gamma2,
                                     n_elec=k)
    return ci, net


@pytest.mark.parametrize("primitive", [
    "rdm1", "rdm2", "hole_rdm1", "pair_matrix", "hole_pair_matrix",
    "koopmans_annihilation", "koopmans_creation",
    "pair_koopmans", "hole_pair_koopmans", "excitation_overlap", "excitation_koopmans",
])
def test_every_served_primitive_matches_the_ci_provider(providers, primitive):
    ci, net = providers
    a = getattr(ci, primitive)()
    b = getattr(net, primitive)()
    assert a.shape == b.shape
    assert np.max(np.abs(a - b)) < TOL, primitive


def test_e_active_and_shifted_counts_match(providers):
    ci, net = providers
    assert abs(ci.e_active - net.e_active) < TOL
    for delta in (-2, -1, 0, 1, 2):
        assert net.shifted_ndet(delta) == ci.shifted_ndet(delta)


def test_the_unserved_primitives_are_absent_or_refuse(providers):
    _, net = providers
    assert not hasattr(net, "annihilation_perturbers")
    assert not hasattr(net, "creation_perturbers")
    with pytest.raises(NotImplementedError):
        net.rdm3()
    with pytest.raises(NotImplementedError):
        net.contract_rdm4(None)


def test_koopmans_gram_refuses_an_odd_string(providers):
    from kuiva.dmrg.density import koopmans_gram
    _, net = providers
    with pytest.raises(ValueError, match="even ladder strings"):
        koopmans_gram(net._ttno_h, net.state, 0,
                      [fermion_term(1.0, [(0, False)])], 0.0)


# --- end-to-end: the shared driver over the two engines --------------------------------------

@pytest.fixture(scope="module")
def lih():
    from test_nevpt2 import lih_setup
    return lih_setup()


@pytest.fixture(scope="module")
def both_routes(lih):
    states = casci(lih["factors"], lih["h_ao"], lih["coeff"], lih["spaces"],
                   lih["nelecas"], n_states=1, e_nuc=lih["e_nuc"], report=False)
    reference = sc_nevpt2(lih["factors"], lih["h_ao"], lih["coeff"], lih["spaces"],
                          states.vectors, lih["nelecas"], energies=states.energies,
                          e_nuc=lih["e_nuc"], report=False)
    ints = CASIntegrals.build(lih["factors"], lih["h_ao"], lih["coeff"], lih["spaces"],
                              e_nuc=lih["e_nuc"])
    solver = DMRGSolver(lih["nelecas"], max_bond=200, n_roots=1, enforce_kramers=False,
                        seed=5)
    solver.solve(ints)
    network = sc_nevpt2_dmrg(lih["factors"], lih["h_ao"], lih["coeff"], lih["spaces"],
                             solver, lih["nelecas"], e_nuc=lih["e_nuc"], report=False)
    return reference, network


def test_the_six_served_classes_agree_class_by_class(both_routes):
    """Same orbitals, same reference state, two contraction machineries: to rounding."""
    reference, network = both_routes
    for name in SERVED:
        a = float(reference.class_energies[name][0])
        b = float(network.class_energies[name][0])
        assert abs(a - b) < TOL, name
    assert abs(float(reference.e_casscf[0]) - float(network.e_casscf[0])) < TOL


def test_the_network_e2_is_partial_and_says_so(both_routes, kuiva_caplog):
    reference, network = both_routes
    assert reference.complete
    assert not network.complete
    assert network.missing == MISSING
    # the partial total is exactly the sum of the served classes
    served_sum = sum(float(network.class_energies[n][0]) for n in SERVED)
    assert abs(float(network.e2[0]) - served_sum) < 1e-12
    # and it equals the CI's sum over the same six classes, not the CI's full E2
    ci_six = sum(float(reference.class_energies[n][0]) for n in SERVED)
    assert abs(float(network.e2[0]) - ci_six) < TOL


def test_the_skipped_classes_warn_at_evaluation(lih, kuiva_caplog):
    ints = CASIntegrals.build(lih["factors"], lih["h_ao"], lih["coeff"], lih["spaces"],
                              e_nuc=lih["e_nuc"])
    solver = DMRGSolver(lih["nelecas"], max_bond=200, n_roots=1, enforce_kramers=False,
                        seed=6)
    solver.solve(ints)
    sc_nevpt2_dmrg(lih["factors"], lih["h_ao"], lih["coeff"], lih["spaces"], solver,
                   lih["nelecas"], e_nuc=lih["e_nuc"], report=False)
    messages = [r.getMessage() for r in kuiva_caplog.records]
    assert any("SKIPPED" in m and "Sr" in m for m in messages)
    assert any("PARTIAL" in m for m in messages)


def test_an_unconverged_solver_is_refused():
    solver = DMRGSolver(2, max_bond=8)
    with pytest.raises(ValueError, match="converged a state"):
        sc_nevpt2_dmrg(None, None, None, None, solver, 2)
