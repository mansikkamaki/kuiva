"""Tier-0 tests for the energy window on the tensor network (``kuiva/dmrg/window.py``).

The rule itself is tested on synthetic spectra in ``test_state_window.py`` and is not
re-tested here; what these check is the three things the network route adds, each against a
target that is exact by construction:

* **the ladder finds the count the exact spectrum has.** The two-fragment oracle's CAS(2, 6)
  is small enough to diagonalize densely, so for a grid of cutoffs the count the network
  resolves to is compared with the count the *same rule* reads off ``eigvalsh`` — the check
  a network resolution can be given at all, and the reason a Tier-0 system with an exact
  spectrum is the one to run it on;
* **the witness is a converged network root**, so the resolved count carries a gap to a root
  the average does not use, and the resolution says which kind of witness it used;
* **a root count the network cannot represent is refused, not truncated.** The narrowest
  two-site window of the tour bounds the ensemble whatever the bond dimension is, and a
  window that would need more roots than that is a window this topology cannot answer.
"""
import numpy as np
import pytest

from kuiva.dmrg import DMRGSolver, NetworkGraph, TTNOTemplate
from kuiva.dmrg.plan import two_site_capacity
from kuiva.dmrg.sweep import random_state
from kuiva.dmrg.window import (PILOT_CAP, WITNESS_ROOTS, pilot_estimate,
                               resolve_network_window, truncate_roots)
from kuiva.util.window import EnergyWindow, resolve_window

from test_dmrg_reconnect import exact_energies, two_fragments

#: The model's level spacings are of order 1 Eh, so the cutoffs are stated in Eh — the unit
#: axis of the request exists exactly so a synthetic spectrum can be asked about in its own
#: units instead of through a conversion nobody reads.
EH = dict(unit="Eh", gap_unit="Eh")


def _window(cutoff, gap=0.2, **kwargs):
    return EnergyWindow(cutoff, manifold_gap=gap, max_states=16, **EH, **kwargs)


@pytest.fixture(scope="module")
def system():
    """``(ttno, n_elec, exact spectrum)`` on the coarse two-node partition.

    Three modes per node deliberately: the two-site window is then the whole CAS(2, 6)
    sector, so this fixture can ask about every root and the capacity refusal below gets a
    system of its own."""
    n, h, eri, _, _ = two_fragments()
    graph = NetworkGraph(2, [(0, 1)], contents=[(0, 1, 2), (3, 4, 5)])
    ttno = TTNOTemplate(graph).fill(h, eri)
    return ttno, 2, exact_energies(n, 2, h, eri, 15)


def test_the_ladder_resolves_the_count_the_exact_spectrum_has(system):
    """⚠ The check that gives a network resolution a target at all. The rule is one
    implementation, so running it on ``eigvalsh`` of the same Hamiltonian and on the
    network's own spectrum tests the *ladder* — the first rung, the growth, the witness and
    the stopping — and nothing else."""
    ttno, n_elec, exact = system
    for cutoff in (1.0, 3.0, 3.2, 6.0, 9.0):
        window = _window(cutoff)
        expected = resolve_window(exact, window, space_size=exact.size)
        resolved = resolve_network_window(ttno, n_elec, window, max_bond=16, report=False)
        assert resolved.resolution.count == expected.count, cutoff
        assert resolved.resolution.complete
        # the energies the verdict was read from are the exact ones at a saturated cap
        m = resolved.resolution.count
        assert np.max(np.abs(resolved.resolution.energies[:m] - exact[:m])) < 1e-8
        # and the production ensemble is the resolved count, witness roots dropped
        assert resolved.state.n_roots == expected.count


def test_the_witness_is_a_converged_network_root(system):
    """A rung solves the count *plus a whole pair* to convergence, so the gap the verdict
    reports is between two converged roots — which is what makes it a statement that the
    window is complete rather than only that it is not obviously incomplete."""
    ttno, n_elec, exact = system
    resolved = resolve_network_window(ttno, n_elec, _window(1.0), max_bond=16, report=False)
    assert resolved.resolution.witness == "converged network roots"
    assert resolved.resolution.count == 1
    gap = resolved.resolution.verdict.witness_gap_eh
    assert abs(gap - (exact[1] - exact[0])) < 1e-8
    # the rung carried the witness pair above the count it was testing
    assert resolved.resolution.energies.size >= resolved.resolution.count + WITNESS_ROOTS
    assert resolved.sweeps and all(s > 0 for s in resolved.sweeps)


def test_growth_climbs_the_ladder_from_a_stated_first_rung(system):
    """``initial=`` overrides every estimate, so the rungs below are the growth rule itself:
    x1.5 on this route, because a rung is a whole sweep campaign rather than a Davidson
    solve."""
    ttno, n_elec, _ = system
    resolved = resolve_network_window(ttno, n_elec, _window(3.0, initial=1), max_bond=16,
                                      report=False)
    rungs = [(r.n_roots, r.verdict.complete) for r in resolved.resolution.rungs]
    assert rungs == [(1, False), (2, False), (3, False), (5, True)]
    assert resolved.resolution.count == 5
    assert resolved.resolution.first_rung_source == "initial="
    assert resolved.pilot is None                    # a stated rung needs no pilot


def test_the_pilot_supplies_the_first_rung_and_says_what_it_cost(system):
    """The network's own cheap estimate: a short campaign at a small cap, read by the rule
    for the first rung only. ⚠ A rung, never a verdict — the pilot's splittings are rough by
    construction and the production spectrum is what the count is finally read from."""
    ttno, n_elec, _ = system
    window = _window(3.0)
    estimate = pilot_estimate(ttno, n_elec, window, report=False)
    assert estimate.cap == PILOT_CAP and estimate.count >= 1
    resolved = resolve_network_window(ttno, n_elec, window, max_bond=16, report=False)
    assert resolved.pilot is not None
    assert resolved.resolution.first_rung_source.startswith("the pilot sweep")
    assert resolved.resolution.rungs[0].n_roots == resolved.pilot.count
    assert "pilot" in resolved.resolution.extra


def test_an_upstream_estimate_is_used_instead_of_a_pilot(system):
    """An upstream cheap CI has already looked at a spectrum, so the pilot is not paid for."""
    ttno, n_elec, _ = system
    resolved = resolve_network_window(ttno, n_elec, _window(3.0), max_bond=16, estimate=4,
                                      report=False)
    assert resolved.pilot is None
    assert resolved.resolution.rungs[0].n_roots == 4
    assert resolved.resolution.count == 5


def test_a_window_the_topology_cannot_hold_is_refused(system):
    """⚠ The narrowest two-site window of the tour bounds the ensemble whatever the cap is,
    and the bound is the **charge sectors** the block can reach, not the bond dimension. On
    the six-node path of this system two adjacent one-spinor nodes reach four sectors, so a
    cutoff whose count plus a witness pair needs more than four roots has no converged
    witness and the window is refused with the capacity, the sector and the fix named —
    never rounded down to the roots the network happens to be able to hold.

    The coarse two-node partition of the same system reaches fifteen (the whole sector),
    which is why the fixture above answers the same request."""
    n, h, eri, _, _ = two_fragments()
    graph = NetworkGraph.path(6)
    ttno = TTNOTemplate(graph).fill(h, eri)
    assert two_site_capacity(graph, 2) == 4
    assert two_site_capacity(NetworkGraph(2, [(0, 1)], contents=[(0, 1, 2), (3, 4, 5)]),
                             2) == 15
    with pytest.raises(ValueError, match="coarser node partition"):
        resolve_network_window(ttno, 2, _window(12.0), max_bond=16, report=False)
    # a cutoff whose count and witness fit inside the capacity is answered on the same
    # topology: one state, with the pair above it as the witness
    resolved = resolve_network_window(ttno, 2, _window(1.0), max_bond=16, report=False)
    assert resolved.resolution.count == 1 and resolved.resolution.complete


def test_a_cap_that_truncates_a_bond_below_a_rung_is_refused_as_the_ladder(system):
    """⚠ The structural ceiling is computed over the **full allowed** sector set, which a
    capped state need not have kept — so a bond dimension small enough to truncate a bond can
    still put a rung past what that bond holds, below the ceiling. The sweep refuses (it
    always did); what this pins is that the refusal comes back saying the roots were a
    *rung's*, not a count anybody stated, since the fix is different for the two."""
    from kuiva.dmrg import TwoSiteCapacityError

    n, h, eri, _, _ = two_fragments()
    graph = NetworkGraph.path(6)
    ttno = TTNOTemplate(graph).fill(h, eri)
    assert two_site_capacity(graph, 2) == 4            # the charge sectors allow four
    # ... but at a cap of two the state carries only three of them, and a rung that asks for
    # the fourth meets the sweep's own refusal rather than the ladder's ceiling
    with pytest.raises(TwoSiteCapacityError, match="rung of an energy window"):
        resolve_network_window(ttno, 2, _window(3.0, gap=0.05, initial=3), max_bond=2,
                               report=False)


def test_warm_growth_reaches_the_same_verdict_as_cold(system):
    """The measured-only variant: a grown rung started from the previous rung's ensemble
    padded with random centers. ⚠ What it may change is the number of sweeps; what it may not
    change is the answer — so the verdict, the count and the spectrum are pinned against the
    cold ladder on the same system."""
    ttno, n_elec, exact = system
    window = _window(3.0, initial=1)
    cold = resolve_network_window(ttno, n_elec, window, max_bond=16, report=False)
    warm = resolve_network_window(ttno, n_elec, window, max_bond=16, warm_growth=True,
                                  report=False)
    assert warm.resolution.count == cold.resolution.count == 5
    assert [r.n_roots for r in warm.resolution.rungs] == [r.n_roots
                                                          for r in cold.resolution.rungs]
    m = warm.resolution.count
    assert np.max(np.abs(warm.resolution.energies[:m] - exact[:m])) < 1e-8


def test_pad_roots_keeps_the_incumbent_basis_and_appends_centers(system):
    ttno, n_elec, _ = system
    from kuiva.dmrg.window import pad_roots

    state = random_state(ttno, n_elec, 8, n_roots=2, rng=np.random.default_rng(2))
    grown = pad_roots(state, 5, np.random.default_rng(3))
    assert grown.n_roots == 5 and state.n_roots == 2      # the original is untouched
    assert grown.graph == state.graph and grown.center == state.center
    for a, b in zip(grown.centers, state.centers):        # the incumbent roots are kept
        assert np.max(np.abs(a.to_dense() - b.to_dense())) == 0.0
    # and asking for fewer is the truncation, not an error
    assert pad_roots(state, 1, np.random.default_rng(4)).n_roots == 1


def test_a_rung_is_a_spectrum_probe_and_the_rdm_gate_does_not_refuse_it():
    """⚠ A rung builds no RDMs, so the state-averaging gate — which is about RDMs — may not
    refuse it. Measured on the first Tier-3 system: at a truncating cap every Kramers pair of
    a 16-root trial ensemble is split, the gate raised, and the window could not be resolved
    at all. What the splitting means is *reported* instead, by
    :func:`~kuiva.dmrg.window.kramers_split_cm`, which is the statement a user can act on."""
    from kuiva.dmrg.window import kramers_split_cm

    # the diagnostic itself: an odd electron count, pairs split by the truncation
    e = np.array([0.0, 1.0e-5, 3.0e-3, 3.0e-3 + 2.0e-5])
    split = kramers_split_cm(e, 4, n_elec=3)
    assert abs(split - 2.0e-5 * 219474.6313632) < 1e-6
    assert kramers_split_cm(e, 4, n_elec=4) is None      # even N: no theorem, no claim
    assert kramers_split_cm(e, 1, n_elec=3) is None      # one root has no pair

    # and a rung that splits pairs resolves rather than raising: an odd-electron model whose
    # truncation cannot hold the ensemble exactly
    n, h, eri, _, _ = two_fragments()
    graph = NetworkGraph(2, [(0, 1)], contents=[(0, 1, 2), (3, 4, 5)])
    ttno = TTNOTemplate(graph).fill(h, eri)
    resolved = resolve_network_window(ttno, 3, _window(3.0), max_bond=4, report=False)
    assert resolved.resolution.count >= 1
    assert "kramers_split_cm" in resolved.resolution.extra


def test_truncate_roots_keeps_the_shared_basis_and_drops_the_witness(system):
    ttno, n_elec, _ = system
    state = random_state(ttno, n_elec, 8, n_roots=4, rng=np.random.default_rng(1))
    kept = truncate_roots(state, 2)
    assert kept.n_roots == 2 and state.n_roots == 4          # the original is untouched
    assert kept.graph == state.graph and kept.center == state.center
    with pytest.raises(ValueError, match="cannot keep"):
        truncate_roots(state, 5)


# --- the solver's side of it ----------------------------------------------------------------

class FragmentInts:
    """A ``CASIntegrals``-shaped stub over the two-fragment oracle (as in
    ``test_dmrg_solver.py``)."""

    e_core = 0.25

    def __init__(self):
        self.n, self.h, self.eri, _, _ = two_fragments()

    def h_active_effective(self):
        return self.h

    def active_eri(self):
        return self.eri


def coarse_graph():
    return NetworkGraph(2, [(0, 1)], contents=[(0, 1, 2), (3, 4, 5)])


def test_an_unresolved_solver_refuses_to_solve_or_name_its_chart():
    solver = DMRGSolver(2, max_bond=16, n_roots=_window(3.0), graph=coarse_graph())
    assert solver.n_roots is None and solver.n_states is None
    assert solver.window == _window(3.0)
    with pytest.raises(RuntimeError, match="has not been resolved"):
        solver.solve(FragmentInts())
    with pytest.raises(RuntimeError, match="has not been resolved"):
        solver.space_key()


def test_resolve_window_returns_a_sibling_at_the_resolved_count():
    """The root count is part of the chart, so a new count is a new solver — and the sibling
    carries the window as provenance, which is what puts the cutoff beside the count in the
    checkpoint's state-average key."""
    from kuiva.io.checkpoint import state_average_key, state_average_window

    ints = FragmentInts()
    window = _window(3.0)
    solver = DMRGSolver(2, max_bond=16, n_roots=window, graph=coarse_graph(),
                        enforce_kramers=False, seed=1)
    assert state_average_key(solver) is None            # nothing to compare yet
    resolution, sibling = solver.resolve_window(ints, report=False)
    assert resolution.count == 5 and sibling.n_roots == 5
    assert sibling.window == window
    assert sibling is not solver and solver.n_roots is None
    key = state_average_key(sibling)
    assert key.startswith("n_states=5")
    assert state_average_window(key) == window
    # the sibling is warm: the ladder's own converged ensemble, truncated to the count
    energy, gamma, _ = sibling.solve(ints)
    exact = exact_energies(ints.n, 2, ints.h, ints.eri, 5)
    assert abs(energy - (float(np.mean(exact)) + ints.e_core)) < 1e-8
    assert abs(np.trace(gamma).real - 2.0) < 1e-8


def test_a_window_refuses_weights_and_the_per_irrep_form():
    with pytest.raises(ValueError, match="weights= cannot be combined"):
        DMRGSolver(2, max_bond=8, n_roots=_window(3.0), weights=[0.5, 0.5])
    with pytest.raises(ValueError, match="per-irrep energy window"):
        DMRGSolver(2, max_bond=8, n_roots={"A": _window(3.0)})


def test_the_estimate_is_carried_so_the_pilot_is_paid_for_once():
    """A later round re-resolves at its own orbitals; the pilot campaign is a property of the
    calculation, not of the round, so the sibling inherits what the first one learned."""
    ints = FragmentInts()
    solver = DMRGSolver(2, max_bond=16, n_roots=_window(3.0), graph=coarse_graph(),
                        enforce_kramers=False, seed=1)
    first, sibling = solver.resolve_window(ints, report=False)
    assert sibling._estimate == 5
    again, third = sibling.resolve_window(ints, n_prev=5, report=False)
    assert again.count == 5 and third.n_roots == 5
    # the pilot is reported by every resolution of the calculation and RUN by one of them
    assert again.extra["pilot"] == first.extra["pilot"]
    assert again.first_rung_source == "the previous round's count plus a witness pair"
