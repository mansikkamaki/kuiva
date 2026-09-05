"""Tier-0 tests for ``DMRGSolver`` — the solver contract and the CASSCF hookup.

Two claims, tested separately:

* **Parity**: driven by the *unchanged* orbital optimizer, the DMRG solver at saturated
  bond dimension reproduces the exact-CI CASSCF trajectory (same iterations, same
  energy) — the ci_solver contract really is solver-agnostic.
* **The contract**: ``solve`` holds its chart, ``propose`` evaluates a re-adapted network
  without touching the incumbent, ``adopt`` commits it, ``space_key`` changes exactly
  then — on the two-fragment oracle, where reconnection has real structure to
  find.

⚠ ``kuiva.dmrg`` never imports ``kuiva.mcscf``; this test imports both and asserts the
protocol *structurally* (``isinstance`` against the runtime-checkable protocol), which is
exactly how the drivers consume the solver.
"""
import numpy as np
import pytest

from kuiva.dmrg import DMRGSolver, NetworkGraph
from kuiva.dmrg.reconnect import ReconnectionPolicy
from kuiva.integrals.transform import ThreeIndexAO
from kuiva.mcscf.adaptive import AdaptiveCISolver, SolverFailure, as_adaptive_solver
from kuiva.mcscf.casci import FullCISolver
from kuiva.mcscf.events import optimize_orbitals_events
from kuiva.mcscf.orbopt import OrbitalSpaces, optimize_orbitals

from test_dmrg_reconnect import exact_energies, two_fragments, wrong_seed_tree


@pytest.fixture(scope="module")
def system():
    """The converging synthetic system of ``test_mcscf.py`` / ``test_orbopt_events.py``:
    CAS(2,4) from a core guess, seed for seed."""
    rng = np.random.default_rng(3)
    nao = 6
    n = 2 * nao
    npair = nao * (nao + 1) // 2
    factors = ThreeIndexAO(l_packed=rng.standard_normal((3 * nao, npair)), nao=nao,
                           origin="cholesky")
    h_ao = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    h_ao = 0.5 * (h_ao + h_ao.conj().T)
    _, c0 = np.linalg.eigh(h_ao)
    return factors, h_ao, np.ascontiguousarray(c0), OrbitalSpaces.from_counts(2, 4, n)


class FragmentInts:
    """A ``CASIntegrals``-shaped stub over the two-fragment oracle."""

    e_core = 0.25

    def __init__(self):
        self.n, self.h, self.eri, _, _ = two_fragments()

    def h_active_effective(self):
        return self.h

    def active_eri(self):
        return self.eri


# --- the contract ---------------------------------------------------------------------------

def test_dmrg_solver_satisfies_the_protocol_structurally():
    solver = DMRGSolver(2, max_bond=8)
    assert isinstance(solver, AdaptiveCISolver)
    assert as_adaptive_solver(solver) is solver          # passed through unwrapped


def test_static_solver_never_proposes():
    ints = FragmentInts()
    solver = DMRGSolver(2, max_bond=16, enforce_kramers=False,
                        graph=wrong_seed_tree(), adaptive=False, seed=4)
    solver.solve(ints)
    assert solver.propose(ints) is None


def test_propose_does_not_touch_the_incumbent_and_adopt_commits():
    ints = FragmentInts()
    ref = exact_energies(ints.n, 2, ints.h, ints.eri, 1)[0] + ints.e_core
    # entropy rule explicitly: at saturated bond dimension nothing is discarded and the
    # solver's weight default (right for capped production runs, measured) is
    # deliberately inert — the reconnection measurement, relied on here in reverse
    solver = DMRGSolver(2, max_bond=16, enforce_kramers=False,
                        graph=wrong_seed_tree(), adaptive=True, seed=4,
                        policy=ReconnectionPolicy(rule="entropy"))
    e0, gamma0, _ = solver.solve(ints)
    assert abs(e0 - ref) < 1e-8                          # saturated D: exact either way
    key0 = solver.space_key()

    proposal = solver.propose(ints)
    assert proposal is not None
    assert proposal.key != key0
    assert solver.space_key() == key0                    # nothing adopted yet
    e_again, gamma_again, _ = solver.solve(ints)         # incumbent chart untouched
    assert abs(e_again - e0) < 1e-9
    assert np.max(np.abs(gamma_again - gamma0)) < 1e-8

    solver.adopt(proposal.key)
    assert solver.space_key() != key0
    assert solver.n_adoptions == 1
    e1, gamma1, _ = solver.solve(ints)
    assert abs(e1 - ref) < 1e-8                          # same physics, new chart
    assert np.max(np.abs(gamma1 - gamma0)) < 1e-7


def test_adopt_without_matching_proposal_is_refused():
    solver = DMRGSolver(2, max_bond=8)
    with pytest.raises(ValueError, match="no pending proposal"):
        solver.adopt("some-key")


def test_unconverged_solve_raises_solver_failure():
    ints = FragmentInts()
    solver = DMRGSolver(2, max_bond=2, enforce_kramers=False, max_sweeps=1,
                        graph=wrong_seed_tree(), seed=4)
    with pytest.raises(SolverFailure):
        solver.solve(ints)


def test_energy_includes_e_core_and_matches_exact_ci():
    ints = FragmentInts()
    solver = DMRGSolver(2, max_bond=16, enforce_kramers=False, seed=1)
    e, gamma, gamma2 = solver(ints)                      # the ci_solver facade
    ref = exact_energies(ints.n, 2, ints.h, ints.eri, 1)[0] + ints.e_core
    assert abs(e - ref) < 1e-8
    assert abs(np.trace(gamma).real - 2.0) < 1e-8


def test_rdms_off_skips_the_extraction_and_keeps_the_transition_densities():
    """A fixed-orbital use gets the same energy with no state-averaged RDMs built — the
    per-node dE/dW extraction is what the memory plan refuses on fat and branching nodes,
    and a network CASCI never reads it — while the transition-density route still serves
    the property matrices."""
    ints = FragmentInts()
    full = DMRGSolver(2, max_bond=16, enforce_kramers=False, seed=1)
    e_full, gamma, _ = full(ints)
    lean = DMRGSolver(2, max_bond=16, enforce_kramers=False, seed=1, rdms=False)
    e, g, g2 = lean(ints)
    assert g is None and g2 is None
    assert abs(e - e_full) < 1e-10
    tdm = lean.transition_densities()
    assert tdm.shape[-2:] == gamma.shape
    assert np.allclose(tdm[0, 0], gamma, atol=1e-8)


# --- the CASSCF hookup ----------------------------------------------------------------------

def test_dmrg_casscf_reproduces_ci_casscf_trajectory(system):
    # ten macro-iterations of the same driver from the same start: the exact CI and the
    # saturated-bond DMRG must produce the same trajectory — energy per iteration, not
    # just the endpoint. Iteration count kept small: parity is the claim under test,
    # convergence of this system is test_mcscf.py's.
    factors, h_ao, c0, spaces = system
    kw = dict(max_iter=10, mode="second-order", conv_grad=1e-5, report=False)
    ci = FullCISolver(4, 2, n_states=2, enforce_kramers=False)
    r_ci = optimize_orbitals(factors, h_ao, c0, spaces, ci, **kw)
    dmrg = DMRGSolver(2, max_bond=64, n_roots=2, enforce_kramers=False, seed=2)
    r_dm = optimize_orbitals(factors, h_ao, c0, spaces, dmrg, **kw)
    assert len(r_dm.history) == len(r_ci.history)
    assert np.max(np.abs(np.asarray(r_dm.history) - np.asarray(r_ci.history))) < 1e-8
    assert np.max(np.abs(r_dm.gamma - r_ci.gamma)) < 1e-8


def test_event_gated_driver_accepts_the_dmrg_solver(system):
    # the event controller drives the DMRG solver unmodified; a non-adaptive chart is
    # simply an event stream of refusals, and the run must still converge on the chart
    factors, h_ao, c0, spaces = system
    solver = DMRGSolver(2, max_bond=8, n_roots=1, enforce_kramers=False, seed=2)
    result = optimize_orbitals_events(factors, h_ao, c0, spaces, solver,
                                      max_iter=40, mode="second-order",
                                      conv_grad=1e-4, report=False)
    assert result.converged
    assert result.event_stable
    assert result.n_adoptions == 0                       # nothing to adopt: fixed chart
    ci = FullCISolver(4, 2, n_states=1, enforce_kramers=False)
    r_ci = optimize_orbitals(factors, h_ao, c0, spaces, ci, max_iter=40,
                             mode="second-order", conv_grad=1e-4, report=False)
    assert abs(result.energy - r_ci.energy) < 1e-7


def test_space_key_is_stable_across_solves_and_names_the_chart():
    ints = FragmentInts()
    solver = DMRGSolver(2, max_bond=16, enforce_kramers=False,
                        graph=wrong_seed_tree(), seed=4)
    solver.solve(ints)
    key = solver.space_key()
    solver.solve(ints)
    assert solver.space_key() == key                     # bond re-derivation is not a
    assert "D16" in key and key.startswith("dmrg")       # chart change (module docstring)


# --- the analysis contract: one_body_moments and transition densities -----------------------

class _PlainInts:
    """Duck-typed integrals over given (h, eri)."""

    e_core = 0.0

    def __init__(self, h, eri):
        self._h, self._eri = h, eri

    def h_active_effective(self):
        return self._h

    def active_eri(self):
        return self._eri


def test_one_body_moments_match_the_ci_route():
    """The network square (per-root 2-RDM route) against the CI square (excitation map).

    The two implementations share nothing — one contracts <a+ a a+ a> through the
    network densities, the other computes ||A|I>||^2 through the determinant excitation
    map — which is what makes the agreement a check rather than a tautology. A random
    spectrum is non-degenerate, so per-state values are basis-independent up to a phase
    that every one of these quantities is invariant under.
    """
    from test_ci_strings import random_spinor_integrals

    n, k, n_roots = 6, 2, 3
    h, eri = random_spinor_integrals(n, seed=41)
    rng = np.random.default_rng(41)
    ops = rng.standard_normal((2, n, n)) + 1j * rng.standard_normal((2, n, n))
    ops = 0.5 * (ops + ops.conj().transpose(0, 2, 1))

    net = DMRGSolver(k, max_bond=200, n_roots=n_roots, enforce_kramers=False, seed=7)
    net.solve(_PlainInts(h, eri))
    ci = FullCISolver(n, k, n_states=n_roots, enforce_kramers=False)
    ref = ci.solve_active(h, eri)
    assert np.max(np.abs(net.last.energies - ref.energies)) < 1e-8

    e_net, sq_net, rdm_net = net.one_body_moments(ops)
    e_ci, sq_ci, rdm_ci = ci.one_body_moments(ops)
    assert np.max(np.abs(e_net - e_ci)) < 1e-8
    assert np.max(np.abs(sq_net - sq_ci)) < 1e-8
    assert np.max(np.abs(rdm_net - rdm_ci)) < 1e-8


def test_one_body_moments_refuse_vectors_and_a_cold_solver():
    solver = DMRGSolver(2, max_bond=8)
    with pytest.raises(RuntimeError, match="solve first"):
        solver.one_body_moments(np.zeros((1, 4, 4)))
    with pytest.raises(ValueError, match="no CI vectors"):
        solver.one_body_moments(np.zeros((1, 4, 4)), vectors=np.zeros((1, 6)))


def test_non_hermitian_operators_are_refused():
    net = DMRGSolver(2, max_bond=16, enforce_kramers=False)
    net.solve(FragmentInts())
    n = FragmentInts().n
    bad = np.zeros((1, n, n), dtype=complex)
    bad[0, 0, 1] = 1.0
    with pytest.raises(ValueError, match="non-Hermitian"):
        net.one_body_moments(bad)


# --- the bond-step ladder: cap changes as chart-change events -------------------------------

def test_bond_steps_climb_through_propose_and_adopt():
    """A mid-run cap change is a chart change: D is in the key and moves only via adopt."""
    from test_ci_strings import random_spinor_integrals

    n, k = 6, 2                                          # even k: random integrals are
    h, eri = random_spinor_integrals(n, seed=51)         # not time-reversal symmetric
    ints = _PlainInts(h, eri)
    solver = DMRGSolver(k, max_bond=200, bond_steps=[3, 200], enforce_kramers=False,
                        seed=3)
    e_low, _, _ = solver.solve(ints)
    assert "D3" in solver.space_key()
    proposal = solver.propose(ints)
    assert proposal is not None
    assert "D200" in proposal.key
    assert proposal.energy <= e_low + 1e-10              # a larger cap is variational
    solver.adopt(proposal.key)
    assert "D200" in solver.space_key()
    e_high, _, _ = solver.solve(ints)
    assert abs(e_high - exact_energies(n, k, h, eri, 1)[0]) < 1e-8
    assert solver.propose(ints) is None                  # ladder exhausted, not adaptive


def test_bond_steps_under_the_event_driver(system):
    # the ladder rides the same event seam the topology proposals use: the driver adopts
    # a rung variationally at fixed integrals and the run converges on the final chart
    factors, h_ao, c0, spaces = system
    solver = DMRGSolver(2, max_bond=12, bond_steps=[2, 12], n_roots=1,
                        enforce_kramers=False, seed=2)
    result = optimize_orbitals_events(factors, h_ao, c0, spaces, solver,
                                      max_iter=60, mode="second-order",
                                      conv_grad=1e-4, report=False)
    assert result.converged
    assert "D12" in solver.space_key()                   # the ladder was climbed
    ci = FullCISolver(4, 2, n_states=1, enforce_kramers=False)
    r_ci = optimize_orbitals(factors, h_ao, c0, spaces, ci, max_iter=40,
                             mode="second-order", conv_grad=1e-4, report=False)
    assert abs(result.energy - r_ci.energy) < 1e-7


def test_bond_steps_validation():
    with pytest.raises(ValueError, match="ascending"):
        DMRGSolver(2, max_bond=16, bond_steps=[16, 8])
    with pytest.raises(ValueError, match="one number"):
        DMRGSolver(2, max_bond=32, bond_steps=[8, 16])
    with pytest.raises(ValueError, match="does not combine"):
        DMRGSolver(2, max_bond=16, bond_steps=[8, 16], restart="nowhere.h5")
    with pytest.raises(ValueError, match="must agree"):
        DMRGSolver(2, max_bond=16, bond_steps=[8, 16], bond_schedule=[4, 16])
