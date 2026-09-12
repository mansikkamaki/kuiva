"""Tier-0 tests for state selection by an energy cutoff (:mod:`kuiva.util.window`).

Every target here is exact by construction, so nothing needs an external reference:

* the **rule** on synthetic spectra, one test per branch — a cutoff on a gap, inside a
  cluster (spill), a chain to the cap (refused, naming the clean counts), the whole space,
  Kramers pairs straddling the cutoff, a sector measured against a global reference, and the
  dead band held and released in both directions;
* the **ladder** over a fake oracle whose spectrum is known, so the rung sequence, the
  growth, the even rounding on an odd-electron system, the cap and the first-rung precedence
  are asserted as arithmetic;
* the **CI solver** on random integrals against a dense diagonalization of the
  independently built Hamiltonian, over a grid of cutoffs — the exit criterion of the stage;
* the **generic witness**, on the spin-free open shell whose lowest-diagonal determinants
  miss an ``Sz`` sector: the ladder with generic vectors finds the right count and spectrum,
  and the control without them does not, so the guard is shown to be load-bearing;
* the **unit table**: round trips, the unchanged wavenumber literal, an unknown unit refused;
* the **refusals**: weights beside a window, a window on a solver that has not been resolved,
  two different windows in one per-irrep request, a CASSCF handed a window while the round
  loop does not exist.
"""
import numpy as np
import pytest

import kuiva
from kuiva.ci.strings import CASSpace, hamiltonian_matrix
from kuiva.integrals.transform import ThreeIndexAO
from kuiva.io.checkpoint import parse_state_average_key, state_average_key
from kuiva.mcscf.casci import (BOUNDARY_WARN_CM, FullCISolver, boundary_report_from_window,
                               casci, casscf, state_average_boundary)
from kuiva.mcscf.orbopt import CASIntegrals, OrbitalSpaces
from kuiva.props import multiplet
from kuiva.rdm.rdm import degenerate_blocks
from kuiva.symm.assign import OrbitalLabels
from kuiva.symm.groups import C2Z
from kuiva.symm.sectors import SectorTable
from kuiva.util import units
from kuiva.util.window import (DEFAULT_FIRST_RUNG, DEFAULT_MANIFOLD_GAP_CM, EnergyWindow,
                               WindowCapReached, chain_blocks, first_rung, next_rung,
                               resolve_states, resolve_states_per_sector, resolve_window,
                               shared_window)
from test_ci_davidson import _open_shell_spin_free_integrals
from test_ci_strings import random_spinor_integrals

CM = 1.0 / units.HARTREE_TO_CM          # one wavenumber in Hartree


def spectrum_cm(*values):
    """A spectrum stated in cm^-1 above the lowest state, returned in Hartree."""
    return np.asarray(values, dtype=float) * CM


# --- units -----------------------------------------------------------------------------------

def test_the_wavenumber_literal_is_unchanged_and_shared():
    """⚠ Every committed reference splitting was converted with this literal."""
    assert units.HARTREE_TO_CM == 219474.6313632
    assert multiplet.HARTREE_TO_CM is units.HARTREE_TO_CM


@pytest.mark.parametrize("unit", units.UNITS)
def test_every_unit_round_trips(unit):
    x = 0.0123456789
    assert units.from_hartree(units.to_hartree(x, unit), unit) == pytest.approx(x, rel=1e-14)


def test_the_codata_factors_are_mutually_consistent():
    # eV <-> K through k_B, kJ/mol <-> kcal/mol through the thermochemical calorie
    assert units.HARTREE_TO_K / units.HARTREE_TO_EV == pytest.approx(11604.518, rel=1e-6)
    assert units.HARTREE_TO_KCAL_MOL * 4.184 == pytest.approx(units.HARTREE_TO_KJ_MOL)
    assert units.HARTREE_TO_MEV == 1000.0 * units.HARTREE_TO_EV


def test_unit_aliases_are_case_insensitive_and_unknown_units_are_refused():
    assert units.canonical_unit("CM-1") == "cm^-1"
    assert units.canonical_unit("wavenumbers") == "cm^-1"
    assert units.canonical_unit("Hartree") == "Eh"
    assert units.canonical_unit("kelvin") == "K"
    with pytest.raises(ValueError, match="unknown energy unit 'furlong'"):
        units.to_hartree(1.0, "furlong")


# --- the request -----------------------------------------------------------------------------

def test_the_window_converts_once_at_construction():
    w = EnergyWindow(1000)
    assert w.unit == "cm^-1" and w.gap_unit == "cm^-1"
    assert w.cutoff_eh == pytest.approx(1000 * CM)
    assert w.gap_eh == pytest.approx(DEFAULT_MANIFOLD_GAP_CM * CM)
    w = EnergyWindow(0.5, "eV", manifold_gap=10.0, gap_unit="meV", initial=4, max_states=32)
    assert w.cutoff_cm == pytest.approx(0.5 * units.HARTREE_TO_CM / units.HARTREE_TO_EV)
    assert w.gap_cm == pytest.approx(0.010 * units.HARTREE_TO_CM / units.HARTREE_TO_EV)
    assert "0.5 eV" in w.describe() and "cm^-1" in w.describe()
    assert EnergyWindow.from_json(w.to_json()) == w


@pytest.mark.parametrize("kwargs,match", [
    (dict(cutoff=-1.0), "positive, finite cutoff"),
    (dict(cutoff=0.0), "positive, finite cutoff"),
    (dict(cutoff=100.0, manifold_gap=0.0), "manifold gap must be positive"),
    (dict(cutoff=100.0, initial=0), "initial="),
    (dict(cutoff=100.0, initial=2.5), "initial="),
    (dict(cutoff=100.0, max_states=0), "max_states="),
    (dict(cutoff=100.0, max_rounds=0), "max_rounds="),
    (dict(cutoff=100.0, unit="furlong"), "unknown energy unit"),
])
def test_an_invalid_window_is_refused_eagerly(kwargs, match):
    with pytest.raises(ValueError, match=match):
        EnergyWindow(**kwargs)


def test_the_default_manifold_gap_is_the_boundary_threshold():
    """⚠ One number, by construction: a resolved cut ends at a gap wider than the manifold
    gap, so it is clean by the boundary diagnostic's own standard."""
    assert DEFAULT_MANIFOLD_GAP_CM == BOUNDARY_WARN_CM == 50.0


def test_the_window_is_exported_at_the_top_level():
    assert kuiva.EnergyWindow is EnergyWindow
    assert "EnergyWindow" in dir(kuiva)


# --- chaining: one implementation ------------------------------------------------------------

@pytest.mark.parametrize("energies,tol", [
    ([0.0, 1e-9, 1.0, 1.0 + 1e-9, 2.0], 1e-6),
    ([0.0, 1.0, 2.0], 1e-6),
    ([], 1e-6),
    (np.sort(np.random.default_rng(1).standard_normal(40)), 0.05),
    (np.cumsum(np.full(30, 40.0)) * CM, 50.0 * CM),         # chains into one block
])
def test_the_rdm_gate_blocks_are_the_chain_blocks(energies, tol):
    assert degenerate_blocks(energies, tol) == chain_blocks(energies, tol)


def test_chain_blocks_refuses_a_descending_spectrum():
    with pytest.raises(ValueError, match="ascending"):
        chain_blocks([1.0, 0.0], 1e-6)


# --- the rule ---------------------------------------------------------------------------------

W = EnergyWindow(1000)


def test_a_cutoff_landing_on_a_gap_takes_everything_below_it():
    e = spectrum_cm(0, 0, 300, 300, 900, 900, 1400, 1400, 2000)
    v = resolve_window(e, W, space_size=100)
    assert v.complete and v.count == 6 and not v.spans_space
    assert v.witness_gap_cm == pytest.approx(500.0)
    assert v.spill_cm == pytest.approx(0.0)
    assert v.edge_cm == pytest.approx((900.0, 1400.0))


def test_a_cutoff_inside_a_manifold_includes_the_manifold_whole():
    """The brief's example: a cluster spanning 995-1005 against a 1000 cm^-1 cutoff is
    included up to 1005, and the spill says by how much the rule extended the window."""
    e = spectrum_cm(0, 0, 995, 1000, 1005, 1400, 1401)
    v = resolve_window(e, W, space_size=100)
    assert v.complete and v.count == 5
    assert v.spill_cm == pytest.approx(5.0)
    assert v.witness_gap_cm == pytest.approx(395.0)


def test_chaining_runs_as_far_as_the_gaps_are_narrow():
    e = spectrum_cm(0, 990, 1030, 1070, 1110, 1150, 1500)
    v = resolve_window(e, W, space_size=100)
    assert v.complete and v.count == 6 and v.spill_cm == pytest.approx(150.0)


def test_every_solved_root_inside_the_window_is_an_incomplete_verdict():
    v = resolve_window(spectrum_cm(0, 100, 200), W, space_size=100)
    assert not v.complete and v.count == 3
    # a chain reaching the last solved root is incomplete too: the witness is not there
    v = resolve_window(spectrum_cm(0, 990, 1030), W, space_size=100)
    assert not v.complete and v.count == 3


def test_the_whole_space_is_complete_without_a_witness():
    """The same reading the boundary diagnostic has: no root left out is a pass, not a gap of
    zero — and the spill still says how far past the cutoff the space reaches."""
    v = resolve_window(spectrum_cm(0, 990, 1030), W, space_size=3)     # 1030 chains to 990
    assert v.complete and v.spans_space and v.count == 3
    assert v.witness_gap_cm is None and v.edge_cm is None
    assert v.spill_cm == pytest.approx(30.0)


def test_a_kramers_pair_straddling_the_cutoff_is_never_split():
    """Pairs realized numerically at 1e-8 Eh (about 2e-3 cm^-1): whichever side of the cutoff
    the pair's centre sits, both members land on the same side of the cut."""
    for centre in (999.999, 1000.0, 1000.001):
        pair = np.array([centre - 1e-3, centre + 1e-3])
        e = np.concatenate([spectrum_cm(0, 0, 500, 500), pair * CM, spectrum_cm(1600, 1600)])
        v = resolve_window(e, W, space_size=100)
        assert v.complete and v.count in (4, 6)
        assert v.count % 2 == 0


def test_the_first_round_has_no_dead_band():
    e = spectrum_cm(0, 0, 990, 990, 1045, 1045, 3000)
    assert resolve_window(e, W).count == 4
    assert not resolve_window(e, W).held


def test_a_state_just_above_the_cutoff_leaving_the_window_is_held():
    """Round 1 counted six (the state was inside); at round 2 it sits 45 cm^-1 above the
    cutoff — within one manifold gap — so the count is held, and the report says both."""
    e = spectrum_cm(0, 0, 990, 990, 1045, 1045, 3000)
    v = resolve_window(e, W, n_prev=6)
    assert v.held and v.count == 6 and v.unheld_count == 4 and v.complete
    assert v.witness_gap_cm == pytest.approx(1955.0)


def test_a_state_just_below_the_cutoff_entering_the_window_is_held():
    e = spectrum_cm(0, 0, 900, 900, 995, 995, 3000)
    v = resolve_window(e, W, n_prev=4)
    assert v.held and v.count == 4 and v.unheld_count == 6


def test_a_state_that_moved_by_more_than_the_gap_releases_the_count():
    e = spectrum_cm(0, 0, 990, 990, 1100, 1100, 3000)         # 100 cm^-1 above: > g
    v = resolve_window(e, W, n_prev=6)
    assert not v.held and v.count == 4
    e = spectrum_cm(0, 0, 900, 900, 940, 940, 3000)           # 60 cm^-1 below: > g
    v = resolve_window(e, W, n_prev=4)
    assert not v.held and v.count == 6


def test_the_dead_band_never_holds_a_count_that_is_no_longer_a_gap_boundary():
    """The previous count now cuts inside a chained manifold; holding it would be exactly the
    cut the window exists to forbid."""
    e = spectrum_cm(0, 0, 990, 990, 1030, 1030, 3000)         # 1030 chains to 990
    v = resolve_window(e, W, n_prev=4)
    assert not v.held and v.count == 6


def test_a_sector_is_measured_against_the_global_lowest_root():
    """The per-irrep form: a sector whose own lowest state is 1200 cm^-1 above the global
    ground state contributes nothing, and one 800 above contributes its states below the cut."""
    reference = 0.0
    v = resolve_window(spectrum_cm(1200, 1250, 3000), W, space_size=50, reference_eh=reference)
    assert v.complete and v.count == 0 and v.edge_cm is None
    v = resolve_window(spectrum_cm(800, 900, 1400), W, space_size=50, reference_eh=reference)
    assert v.complete and v.count == 2


def test_the_rule_refuses_bad_input():
    with pytest.raises(ValueError, match="not ascending"):
        resolve_window(spectrum_cm(0, 500, 200), W)
    with pytest.raises(ValueError, match="roots solved in a space of"):
        resolve_window(spectrum_cm(0, 1, 2), W, space_size=2)
    with pytest.raises(ValueError, match="empty spectrum"):
        resolve_window([], W)


# --- the ladder --------------------------------------------------------------------------------

class FakeOracle:
    """A spectrum oracle over a known exact spectrum, recording every request."""

    def __init__(self, exact):
        self.exact = np.asarray(exact, dtype=float)
        self.calls = []
        self.n_apply = 0

    def spectrum(self, n_roots):
        self.calls.append(int(n_roots))
        self.n_apply += int(n_roots)
        return self.exact[:int(n_roots)]


def test_the_ladder_doubles_from_the_default_first_rung_until_the_rule_answers():
    exact = np.concatenate([spectrum_cm(*range(0, 25 * 30, 30)), spectrum_cm(2000, 2100)])
    oracle = FakeOracle(exact)                             # 25 states chained to 720, then 2000
    r = resolve_states(oracle, W, n_elec=4, space_size=exact.size)
    assert oracle.calls == [8, 16, 27]                     # 32 clamped to the space
    assert r.complete and r.count == 25
    assert [rung.n_roots for rung in r.rungs] == oracle.calls
    assert [rung.verdict.complete for rung in r.rungs] == [False, False, True]
    assert r.first_rung_source == "the default first rung"
    assert r.rungs[0].n_apply == 8 and r.n_apply == 8 + 16 + 27


def test_rungs_are_whole_kramers_pairs_on_an_odd_electron_system():
    exact = spectrum_cm(*np.repeat(np.arange(0, 3000, 100), 2))
    r = resolve_states(FakeOracle(exact), EnergyWindow(1000, initial=7), n_elec=5,
                       space_size=exact.size)
    assert r.rungs[0].n_roots == 8 and r.first_rung_source == "initial="
    assert all(rung.n_roots % 2 == 0 for rung in r.rungs)
    assert r.count == 22                                   # 0..1000 inclusive, 11 pairs


def test_first_rung_precedence_is_initial_then_estimate_then_default():
    assert first_rung(EnergyWindow(100, initial=3), estimate=20) == (3, "initial=")
    assert first_rung(EnergyWindow(100), estimate=20) == (20, "the upstream estimate")
    assert first_rung(EnergyWindow(100)) == (DEFAULT_FIRST_RUNG, "the default first rung")
    # a previous round's count needs a witness pair above it, and the cap clamps everything
    assert first_rung(EnergyWindow(100), n_prev=12)[0] == 14
    assert first_rung(EnergyWindow(100, max_states=10), estimate=20)[0] == 10
    assert next_rung(10, EnergyWindow(100), n_elec=5) == 20
    assert next_rung(10, EnergyWindow(100), n_elec=5, growth=1.5) == 16
    assert next_rung(10, EnergyWindow(100), space_size=13) == 13


def test_reaching_the_cap_without_a_verdict_refuses_and_names_the_clean_counts():
    """⚠ Refused, never rounded: a cap that truncated the chain would cut inside a manifold.
    The message names the counts at which a gap wider than the manifold gap *does* occur, so
    the user can state one of them, lower the cutoff, or raise the cap."""
    exact = spectrum_cm(*([0, 0, 300, 300] + list(range(2000, 2000 + 40 * 30, 30))))
    with pytest.raises(WindowCapReached) as info:
        resolve_states(FakeOracle(exact), EnergyWindow(2500, max_states=32), n_elec=4,
                       space_size=exact.size)
    text = str(info.value)
    assert "chains past its cap of 32 states" in text
    assert "largest unambiguous count under the cap is 4" in text
    assert "wider than the manifold gap does occur: 2, 4" in text
    assert "raise max_states" in text


def test_a_dense_first_rung_asks_for_the_whole_space_once():
    exact = spectrum_cm(0, 0, 400, 400, 1300, 1300)
    oracle = FakeOracle(exact)
    r = resolve_states(oracle, W, n_elec=3, space_size=6, first=(6, "the dense solve"))
    assert oracle.calls == [6] and r.count == 4 and r.first_rung_source == "the dense solve"


def test_the_ladder_report_is_output_grammar_only(kuiva_caplog):
    import logging
    exact = spectrum_cm(0, 0, 400, 400, 1300, 1300, 2000, 2000, 2500, 2600)
    r = resolve_states(FakeOracle(exact), W, n_elec=3, space_size=exact.size)
    with kuiva_caplog.at_level(logging.INFO, logger="kuiva"):
        r.report()
    text = kuiva_caplog.text
    assert "state window" in text and "cutoff" in text and "resolved" in text
    assert "witness gap 900.00 cm^-1" in text
    assert "400.00 | 1300.00" in text
    assert r.to_json() and '"count": 4' in r.to_json()


# --- the per-irrep form over fake oracles -----------------------------------------------------

def test_sectors_are_merged_and_the_rule_runs_against_the_global_lowest_root():
    """Sector A holds the ground state and states at 300 and 2000; sector B's lowest state is
    at 980 with a partner at 1020 chained to it (40 cm^-1 apart) across the cutoff. The merged
    count is 4 with two from each sector — B's own spectrum alone could not have placed the
    cut, since its cutoff is measured from A's ground state — and the tightest edge is B's."""
    a = FakeOracle(spectrum_cm(0, 300, 2000, 2100, 2200, 2300))
    b = FakeOracle(spectrum_cm(980, 1020, 1800, 1900, 2500, 2600))
    r = resolve_states_per_sector({"A": a, "B": b}, {"A": W, "B": W},
                                  sizes={"A": 6, "B": 6}, names={"A": "A", "B": "B"})
    assert r.complete and r.count == 4 and r.counts == {"A": 2, "B": 2}
    assert r.verdict.spill_cm == pytest.approx(20.0)
    assert r.verdict.witness_gap_cm == pytest.approx(780.0)
    assert r.sector == "B"                                 # 1800 - 1020 against A's 2000 - 300
    assert r.extra["tightest_gap_cm"] == pytest.approx(780.0)


def test_a_sector_above_the_window_is_dropped_and_a_fixed_count_rides_beside():
    a = FakeOracle(spectrum_cm(0, 0, 400, 400, 3000, 3000))
    b = FakeOracle(spectrum_cm(1500, 1600, 1700, 1800))
    c = FakeOracle(spectrum_cm(5000, 5100, 5200, 5300))
    r = resolve_states_per_sector({"A": a, "B": b, "C": c}, {"A": W, "B": W, "C": 2},
                                  sizes={"A": 6, "B": 4, "C": 4},
                                  names={"A": "A", "B": "B", "C": "C"})
    assert r.counts == {"A": 4, "C": 2} and r.count == 6
    assert c.calls == [4]                                  # count + a witness pair, once


def test_only_the_sector_bounding_the_trusted_spectrum_is_grown():
    """Sector A's 8 roots reach 350 cm^-1 while B's reach far past the cutoff: nothing above
    350 can be trusted until A has solved past it, and only A is re-solved."""
    a = FakeOracle(spectrum_cm(*np.arange(0, 60 * 50, 50)))
    b = FakeOracle(spectrum_cm(*np.arange(20, 20 + 60 * 400, 400)))
    r = resolve_states_per_sector({"A": a, "B": b}, {"A": W, "B": W},
                                  sizes={"A": 60, "B": 60}, names={"A": "A", "B": "B"})
    assert b.calls == [8] and a.calls[0] == 8 and len(a.calls) > 1
    # every A state up to 1000 chains (50 cm^-1 steps are not > g) and so does 1050 ... up to
    # the first gap wider than g in the merged spectrum
    assert r.complete and r.counts["B"] == 3


def test_two_different_windows_in_one_request_are_refused():
    with pytest.raises(ValueError, match="ONE energy window"):
        shared_window({"A": EnergyWindow(1000), "B": EnergyWindow(500)})
    assert shared_window({"A": EnergyWindow(1000), "B": EnergyWindow(1000), "C": 2}) == W


# --- the CI solver -----------------------------------------------------------------------------

def dense_spectrum(n, k, h, eri):
    return np.linalg.eigvalsh(hamiltonian_matrix(CASSpace(n, k).determinants(), h, eri)
                              .toarray())


@pytest.fixture(scope="module")
def iterative_system():
    """CAS(4, 11 spinors): 330 determinants, above the dense threshold, so the ladder is real
    Davidson solves with warm starts between rungs."""
    n, k = 11, 4
    h, eri = random_spinor_integrals(n, seed=7, scale=0.05)
    return n, k, h, eri, dense_spectrum(n, k, h, eri)


def test_an_unresolved_window_solver_refuses_to_solve_and_names_the_fix():
    solver = FullCISolver(6, 3, n_states=W)
    assert solver.n_states is None and solver.window == W and "unresolved" in repr(solver)
    h, eri = random_spinor_integrals(6, seed=1)
    with pytest.raises(RuntimeError, match="resolve_window"):
        solver.solve_active(h, eri)
    with pytest.raises(RuntimeError, match="resolve_window"):
        solver.spectrum(h, eri, 2)


def test_weights_beside_a_window_are_refused():
    with pytest.raises(ValueError, match="weights= cannot be combined"):
        FullCISolver(6, 3, n_states=W, weights=[0.5, 0.5])
    with pytest.raises(ValueError, match="there is nothing to resolve"):
        FullCISolver(6, 3, n_states=2).resolve_window(*random_spinor_integrals(6, seed=1))


@pytest.mark.parametrize("position", [3, 6, 10, 15, 21])
def test_the_resolved_count_equals_the_dense_spectrum_count(iterative_system, position):
    """The exit criterion: over a grid of cutoffs, the count the ladder resolves equals the
    count the rule reads off ``eigh`` of the independently built Hamiltonian, and the states
    the sibling solver then solves are those lowest states."""
    n, k, h, eri, exact = iterative_system
    rel = (exact - exact[0]) * units.HARTREE_TO_CM
    cutoff = 0.5 * (rel[position - 1] + rel[position])      # between two exact states
    window = EnergyWindow(cutoff)
    expected = resolve_window(exact, window, space_size=exact.size)
    assert expected.complete
    solver = FullCISolver(n, k, n_states=window, enforce_kramers=False)
    resolution, resolved = solver.resolve_window(h, eri, e_core=-1.0, report=False)
    assert resolution.complete and resolution.count == expected.count
    assert resolution.boundary_gap_cm == pytest.approx(expected.witness_gap_cm, abs=1e-6)
    assert resolved.n_states == expected.count and resolved.window == window
    result = resolved.solve_active(h, eri, e_core=-1.0)
    assert np.allclose(result.energies, exact[:expected.count], atol=1e-9)
    assert np.allclose(result.total_energies, exact[:expected.count] - 1.0, atol=1e-9)
    assert result.weights.size == expected.count


def test_the_sibling_shares_the_workspace_and_records_the_window_in_the_key(iterative_system):
    n, k, h, eri, exact = iterative_system
    solver = FullCISolver(n, k, n_states=W, enforce_kramers=False)
    resolution, resolved = solver.resolve_window(h, eri, report=False)
    assert resolved._sigma is solver._sigma and resolved.space is solver.space
    key = state_average_key(resolved)
    assert key.startswith("n_states={};".format(resolution.count))
    assert ";window=" in key and '"cutoff": 1000.0' in key
    n_states, weights = parse_state_average_key(key)
    assert n_states == resolution.count and weights is None
    # the warm start is the ladder's vectors: the first solve is (nearly) free
    result = resolved.solve_active(h, eri)
    assert result.n_iter <= 2


def test_a_small_space_is_resolved_in_one_dense_rung():
    n, k = 8, 3                                            # 56 determinants: dense
    h, eri = random_spinor_integrals(n, seed=3, scale=0.05)
    exact = dense_spectrum(n, k, h, eri)
    window = EnergyWindow(0.5 * (exact[9] - exact[0] + exact[10] - exact[0])
                          * units.HARTREE_TO_CM)
    resolution, resolved = FullCISolver(n, k, n_states=window,
                                        enforce_kramers=False).resolve_window(h, eri,
                                                                              report=False)
    assert len(resolution.rungs) == 1 and resolution.rungs[0].n_roots == 56
    assert "dense" in resolution.first_rung_source
    assert resolution.count == 10 == resolved.n_states


def test_the_window_verdict_is_the_boundary_report(iterative_system):
    """The resolution *is* the boundary measurement: the report built from the verdict agrees
    with the diagnostic measured separately at the same integrals and count."""
    n, k, h, eri, exact = iterative_system
    resolution, resolved = FullCISolver(n, k, n_states=W,
                                        enforce_kramers=False).resolve_window(h, eri,
                                                                              report=False)
    report = boundary_report_from_window(resolution, resolved.ndet, where="fixed orbitals")
    assert report.n_states == resolution.count and report.is_clean
    resolved.solve_active(h, eri)
    ints = CASIntegrals.__new__(CASIntegrals)              # the diagnostic needs h/eri only
    ints.h_active_effective = lambda: h                    # noqa: E731
    ints.active_eri = lambda: eri                          # noqa: E731
    measured = state_average_boundary(resolved, ints, margin=2)
    assert measured.gap_cm == pytest.approx(report.gap_cm, abs=1e-5)


# --- the generic witness is load-bearing -------------------------------------------------------

def test_the_ladder_finds_the_sector_the_biased_guess_misses(monkeypatch):
    """⚠ The mechanism the window makes load-bearing. On a spin-free open shell the natural
    guess can lie entirely in a few ``Sz`` sectors, and a Krylov method never leaves them:
    the eigensolver then converges roots that are **not the lowest** and the rule reads a
    plausible count off a wrong spectrum. The ladder asks for generic vectors at every rung
    (``generic=True``), and this asserts both halves — that it then resolves the exact count
    and spectrum, and that the same ladder *without* the generic vectors does not. A guard
    that cannot fail proves nothing."""
    import importlib
    module = importlib.import_module("kuiva.mcscf.casci")   # the package exports a function

    n_spatial, n_elec = 6, 5
    h, eri = _open_shell_spin_free_integrals(n_spatial, seed=1)
    n = 2 * n_spatial
    exact = dense_spectrum(n, n_elec, h, eri)
    rel = (exact - exact[0]) * units.HARTREE_TO_CM
    # the first clean gap at or past the default first rung, so the ladder's first rung is the
    # one whose 16 lowest-diagonal guesses miss a sector
    clean = [i for i in range(DEFAULT_FIRST_RUNG, 40) if rel[i] - rel[i - 1] > 50.0]
    position = clean[0]
    window = EnergyWindow(0.5 * (rel[position - 1] + rel[position]))
    expected = resolve_window(exact, window, space_size=exact.size)

    resolution, resolved = FullCISolver(n, n_elec, n_states=window).resolve_window(
        h, eri, report=False)
    assert resolution.count == expected.count == position
    assert np.allclose(resolution.energies[:position], exact[:position], atol=1e-8)
    assert np.allclose(resolved.solve_active(h, eri).energies, exact[:position], atol=1e-8)

    # The control: the same ladder with the generic vectors switched off at every rung.
    original = module._CIOracle.spectrum

    def blind(self, n_roots):
        result = self.solver._davidson(self.sigma, self.diagonal, int(n_roots),
                                       guess=self.vectors, label="CAS window",
                                       generic=False)
        self.vectors = result.vectors
        self.n_apply += int(result.n_apply)
        return np.asarray(result.energies[:int(n_roots)], dtype=float) + self.e_core

    monkeypatch.setattr(module._CIOracle, "spectrum", blind)
    try:
        blind_resolution, _ = FullCISolver(n, n_elec, n_states=window).resolve_window(
            h, eri, report=False)
        wrong = (blind_resolution.count != position or not np.allclose(
            blind_resolution.energies[:position], exact[:position], atol=1e-8))
    except WindowCapReached:
        wrong = True                                       # a wrong spectrum that chained
    finally:
        monkeypatch.setattr(module._CIOracle, "spectrum", original)
    assert wrong, "this system no longer exercises the biased-guess failure; pick another seed"


# --- the per-irrep form on the CI solver -------------------------------------------------------

def symmetric_integrals(labels, seed):
    """Random spinor integrals projected onto the sector-conserving pattern of ``labels``."""
    n = len(labels)
    h, eri = random_spinor_integrals(n, seed=seed, scale=0.05)
    lab = np.asarray(labels)
    diff = (lab[:, None] - lab[None, :]) % 4
    h = np.where(diff == 0, h, 0.0)
    total = (diff[:, :, None, None] + diff[None, None, :, :]) % 4
    eri = np.where(total == 0, eri, 0.0)
    return np.ascontiguousarray(h), np.ascontiguousarray(eri)


def test_per_irrep_windows_resolve_against_the_merged_spectrum_and_name_the_tightest_edge():
    labels = [1, 1, 3, 3, 1, 3, 1, 3]                     # C2(z) fermion labels 1E1/2, 2E1/2
    h, eri = symmetric_integrals(labels, seed=11)
    n, k = len(labels), 4
    orbital_labels = OrbitalLabels(group=C2Z, labels=np.asarray(labels)[:, None])
    cas = CASSpace(n, k)
    dense = np.asarray(hamiltonian_matrix(cas.determinants(), h, eri).todense())
    table = SectorTable.build(cas.occupations(), orbital_labels.labels, C2Z)
    sector_spectra = {table.name(t): np.linalg.eigvalsh(dense[np.ix_(table.indices(t),
                                                                     table.indices(t))])
                      for t in table.sectors}
    merged = np.sort(np.concatenate(list(sector_spectra.values())))
    rel = (merged - merged[0]) * units.HARTREE_TO_CM
    position = next(i for i in range(4, 20) if rel[i] - rel[i - 1] > 50.0)
    window = EnergyWindow(0.5 * (rel[position - 1] + rel[position]))
    expected = {name: int(np.count_nonzero(spec <= merged[position - 1] + 1e-12))
                for name, spec in sector_spectra.items()}
    expected = {name: c for name, c in expected.items() if c}

    request = {name: window for name in sector_spectra}
    solver = FullCISolver(n, k, n_states=request, symmetry=orbital_labels,
                          enforce_kramers=False)
    resolution, resolved = solver.resolve_window(h, eri, report=False)
    assert resolution.counts == expected and resolution.count == position
    assert resolution.sector in expected                   # the tightest irrep is named
    result = resolved.solve_active(h, eri)
    assert np.allclose(np.sort(result.energies), merged[:position], atol=1e-8)
    assert resolved.state_request is not None
    # a fixed count beside the window rides along unchanged
    fixed_name, other = sorted(sector_spectra)[0], sorted(sector_spectra)[1]
    mixed = FullCISolver(n, k, n_states={fixed_name: 1, other: window},
                         symmetry=orbital_labels, enforce_kramers=False)
    res2, _ = mixed.resolve_window(h, eri, report=False)
    assert res2.counts[fixed_name] == 1


def test_a_per_irrep_window_without_labels_is_refused():
    with pytest.raises(ValueError, match="needs irrep labels"):
        FullCISolver(8, 4, n_states={"1E1/2": W})


# --- the drivers -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def driver_system():
    rng = np.random.default_rng(3)
    nao = 6
    n = 2 * nao
    npair = nao * (nao + 1) // 2
    factors = ThreeIndexAO(l_packed=rng.standard_normal((3 * nao, npair)), nao=nao,
                           origin="cholesky")
    h_ao = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    h_ao = 0.5 * (h_ao + h_ao.conj().T)
    _, c0 = np.linalg.eigh(h_ao)
    spaces = OrbitalSpaces.from_counts(n_inactive=2, n_active=6, n_orb=n)
    return factors, h_ao, np.ascontiguousarray(c0), spaces, 3


def test_casci_with_a_window_resolves_then_solves_at_the_count(driver_system):
    factors, h_ao, c0, spaces, n_elec = driver_system
    fixed = casci(factors, h_ao, c0, spaces, n_elec, n_states=20, report=False,
                  enforce_kramers=False)
    rel = fixed.excitation_energies_cm()
    position = next(i for i in range(2, 19) if rel[i] - rel[i - 1] > 50.0)
    window = EnergyWindow(0.5 * (rel[position - 1] + rel[position]))
    windowed = casci(factors, h_ao, c0, spaces, n_elec, n_states=window, report=False,
                     enforce_kramers=False)
    assert windowed.window is not None and windowed.window.count == position
    assert windowed.energies.size == position
    assert np.allclose(windowed.energies, fixed.energies[:position], atol=1e-9)
    assert windowed.solver.n_states == position and windowed.solver.window == window


def test_a_casscf_handed_a_window_says_the_round_loop_does_not_exist_yet(driver_system):
    factors, h_ao, c0, spaces, n_elec = driver_system
    with pytest.raises(NotImplementedError, match="not implemented yet"):
        casscf(factors, h_ao, c0, spaces, n_elec, n_states=W, report=False)


def test_the_kramers_restricted_mode_resolves_the_same_window_as_the_general_path():
    """Rungs are pairs and the witness is a pair; the resolved count and the spectrum must be
    the general path's. The spin-free open shell is time-reversal symmetric, which the
    restricted mode checks at every solve."""
    n_spatial, n_elec = 6, 5
    h, eri = _open_shell_spin_free_integrals(n_spatial, seed=1)
    n = 2 * n_spatial
    exact = dense_spectrum(n, n_elec, h, eri)
    rel = (exact - exact[0]) * units.HARTREE_TO_CM
    position = next(i for i in range(DEFAULT_FIRST_RUNG, 40) if rel[i] - rel[i - 1] > 50.0)
    window = EnergyWindow(0.5 * (rel[position - 1] + rel[position]))
    general, _ = FullCISolver(n, n_elec, n_states=window).resolve_window(h, eri, report=False)
    restricted, resolved = FullCISolver(n, n_elec, n_states=window,
                                        kramers="restricted").resolve_window(h, eri,
                                                                             report=False)
    assert restricted.count == general.count == position
    assert all(rung.n_roots % 2 == 0 for rung in restricted.rungs)
    result = resolved.solve_active(h, eri)
    assert resolved.kramers == "restricted" and result.energies.size == position
    assert np.allclose(result.energies, exact[:position], atol=1e-8)
