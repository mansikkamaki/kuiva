"""Tier 0/1: the round loop, the pruning, the budget and the root proposal.

What is asserted, and why each one can fail silently
-----------------------------------------------------
The protocol is a sequence of decisions taken on numbers a *qualitative* probe produced, and
every one of them can go wrong in a way that still yields a perfectly well-formed active
space:

* **a prune may never split a Kramers pair or a degenerate group.** A space is defined on
  spatial orbitals, and keeping some members of a degenerate group selects a subspace that is
  not invariant under whatever made them degenerate -- which for symmetry partners means an
  active space that breaks a degeneracy the molecule has exactly.
* **a fixed class is whole or nothing.** A shell with pairs missing is a different physical
  statement wearing the shell's name; the core over budget is a refusal, and it has to be
  raised *before* the first probe, not after paying for one.
* ⚠ **the decider must be load-bearing.** The keep/drop verdict is the spectrum's, and the
  test that this is not decoration is the one that shows the two rules disagreeing on a real
  system: on ``ti2cl6`` the bridge orbitals are kept by the spectrum and would be dropped by
  the entropy alone. A guard whose two branches never differ is the same defect as one that
  never fires.
* **a manifold boundary is read outward and never inward**, and a boundary that was not found
  is reported as a lower bound rather than rounded to a number.

⚠ **Most of this runs against a scripted probe**, not a real one. What is under test is the
*decision*, and a decision tested through a pre-optimization would be a slow test of a
conditional whose inputs nobody controls -- the real probes are in the Tier-1 tests at the
bottom of the file, where the subject is whether the numbers come out physical.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "generate"))

import systems as sysdef                                                    # noqa: E402

from kuiva.autocas import candidates as cand                                # noqa: E402
from kuiva.autocas import centres as ctr                                    # noqa: E402
from kuiva.autocas import protocol as proto                                 # noqa: E402
from kuiva.autocas import roots as rts                                      # noqa: E402
from kuiva.autocas.probe import ProbeBudget, ProbeResult, probe_roots       # noqa: E402
from kuiva.mcscf.adaptive import SolverFailure                              # noqa: E402
from kuiva.util.window import DEFAULT_MANIFOLD_GAP_CM                       # noqa: E402
from test_autocas_centres import reference_for                              # noqa: E402


# --- the manifold rule (pure) ---------------------------------------------------------------

def test_a_boundary_is_the_first_gap_at_or_above_the_floor():
    """A doublet, then a manifold 300 cm^-1 up: the floor of 2 lands on the doublet and the
    boundary is *there*, not at the next manifold.

    ⚠ The separations matter: two states 10 cm^-1 apart are **one** manifold by the default
    50 cm^-1 rule, so a spectrum written with small spacings would be testing the chaining and
    not the floor.
    """
    spectrum = [0.0, 0.1, 300.0, 300.1, 5000.0, 5000.1]
    boundary = rts.manifold_boundary(spectrum, 2)
    assert boundary.found and boundary.count == 2
    assert boundary.gap_cm == pytest.approx(299.9)


def test_a_floor_inside_a_manifold_is_extended_outward_never_truncated():
    """⚠ The rule that is one-sided on purpose. A four-fold manifold cut in half leaves an even
    remainder and passes every Kramers check, so nothing downstream can catch it -- the
    boundary has to come out at 4 for a floor of 2 and of 3 alike."""
    spectrum = [0.0, 1.0, 2.0, 3.0, 900.0, 901.0]          # one four-fold block, then a pair
    for floor in (1, 2, 3, 4):
        boundary = rts.manifold_boundary(spectrum, floor)
        assert boundary.found and boundary.count == 4, floor
    assert rts.manifold_boundary(spectrum, 5).count == 6


def test_a_spectrum_that_chains_to_its_own_end_reports_a_lower_bound():
    """No gap wider than the manifold gap above the floor: **not found**, and the count is the
    roots solved rather than a number invented at the edge."""
    boundary = rts.manifold_boundary([0.0, 10.0, 20.0, 30.0], 2)
    assert not boundary.found and boundary.gap_cm is None
    assert boundary.count == 4


def test_the_target_spectrum_is_the_manifold_and_its_gap():
    values = rts.target_spectrum([0.0, 5.0, 900.0, 905.0], 2)
    assert list(values[:2]) == [0.0, 5.0]
    assert values[-1] == pytest.approx(895.0)          # the gap to the next manifold


def test_a_change_of_manifold_size_is_not_a_distance():
    """⚠ A class that changed the *structure* of the ground manifold changed more than any
    tolerance measures; the distance is infinite and the reason is in the note."""
    d, note = proto.spectrum_distance([0.0, 5.0, 900.0], [0.0, 5.0, 6.0, 900.0])
    assert not np.isfinite(d) and "changed size" in note
    d, note = proto.spectrum_distance([0.0, 5.0, 900.0], [0.0, 5.0, 880.0])
    assert d == pytest.approx(20.0) and note == ""


# --- the proposal (pure) ---------------------------------------------------------------------

def test_the_proposal_is_a_boundary_and_a_window_that_says_the_same_thing():
    proposal = rts.propose_roots([0.0, 0.0, 1200.0, 1200.0], floor=2)
    assert proposal.n_states == 2
    # The cutoff sits in the middle of the witness gap, so a production ladder re-resolving it
    # against a spectrum that moved lands on the same boundary.
    assert proposal.window.cutoff == pytest.approx(600.0)
    assert proposal.window.cutoff_cm > max(proposal.boundary.gap_cm * 0.0, 0.0)


def test_a_boundary_above_the_cap_proposes_no_count_at_all():
    """⚠ A capped count would be a cut inside the manifold, which is the one thing the whole
    rule exists to prevent -- so no count is proposed and the product is named."""
    spectrum = list(np.linspace(0.0, 40.0, 40)) + [9000.0]
    proposal = rts.propose_roots(spectrum, floor=40, product_floor=256, max_states=16)
    assert proposal.capped and proposal.n_states is None
    assert proposal.window is not None and proposal.window.max_states == 16
    assert "256" in " ".join(proposal.notes)


def test_no_boundary_proposes_the_floor_and_says_it_is_a_lower_bound():
    proposal = rts.propose_roots([0.0, 10.0, 20.0, 30.0], floor=2)
    assert proposal.n_states == 2 and not proposal.boundary.found
    assert "boundary not found" in " ".join(proposal.notes)


def test_a_count_inside_a_manifold_of_the_fixed_orbital_spectrum_is_extended_to_its_end():
    """⚠ The tice_far mechanism, without an SCF. A pre-optimization split a 24-fold manifold
    into three blocks of eight over 196 cm^-1 while the construction orbitals show it within
    5.4 cm^-1; read off the probe alone the count is 8, a cut inside the manifold. Held to the
    fixed-orbital spectrum it is 24, the probe's own gap is where the window goes, and the
    note says why. Where the probe's boundary is also a fixed-orbital one nothing moves."""
    probe = [0.0] * 8 + [98.0] * 8 + [196.0] * 8 + [780.0] * 8
    fixed = [0.0] * 20 + [5.4] * 4 + [779.0] * 8
    plain = rts.propose_roots(probe, 6, product_floor=12)
    held = rts.propose_roots(probe, 6, product_floor=12, fixed_spectrum_cm=fixed,
                             fixed_dimension=276)
    assert plain.n_states == 8
    assert held.n_states == 24 and held.boundary.gap_cm == pytest.approx(584.0)
    assert held.window.cutoff == pytest.approx(488.0)
    assert any("extended to 24" in n for n in held.notes)
    agree = rts.propose_roots([0.0, 0.0, 900.0, 900.0], 2, fixed_spectrum_cm=[0.0, 0.0, 850.0],
                              fixed_dimension=10)
    assert agree.n_states == 2 and not agree.notes


def test_a_count_beyond_what_the_fixed_spectrum_solved_is_not_held_to_it():
    """The fixed-orbital spectrum constrains only the counts it has seen a state above."""
    probe = [0.0] * 4 + [500.0] * 4 + [1500.0] * 2
    short = [0.0] * 4 + [3.0] * 2                   # solved six roots, no boundary seen
    held = rts.propose_roots(probe, 2, fixed_spectrum_cm=short, fixed_dimension=100)
    assert held.n_states == 8


def test_the_probe_averages_a_count_that_ends_on_a_fixed_orbital_boundary(monkeypatch):
    """``floor + margin`` roots extended to the first fixed-orbital boundary past the ground
    manifold, with the CI re-solved for more roots until one is visible -- and left alone
    where it already is one, or where the space is exhausted."""
    from types import SimpleNamespace

    ladder = [0.0] * 24 + [779.0] * 36 + [2400.0] * 40
    calls = []

    def fake_measure(reference, coeff, space, *, n_roots, budget=None, seed=None):
        calls.append(n_roots)
        return SimpleNamespace(spectrum_cm=np.asarray(ladder[:n_roots]))

    monkeypatch.setattr(proto, "measure", fake_measure)
    space = SimpleNamespace(n_active=24, n_elec=2)
    decided = SimpleNamespace(spectrum_cm=np.asarray(ladder[:20]))
    count, note, fixed = proto._probe_count_on_a_boundary(
        None, None, space, decided, n_roots=20, floor=12, gap_cm=50.0, max_states=64,
        budget=None)
    assert count == 60 and "averages 60 roots" in note
    assert calls == [40, 80] and fixed.spectrum_cm.size == 80

    calls.clear()
    whole = SimpleNamespace(spectrum_cm=np.asarray([0.0, 0.0, 900.0, 900.0, 950.0, 950.0,
                                                    3000.0, 3000.0, 3100.0, 3100.0]))
    count, note, _ = proto._probe_count_on_a_boundary(
        None, None, SimpleNamespace(n_active=10, n_elec=1), whole, n_roots=10, floor=2,
        gap_cm=50.0, max_states=64, budget=None)
    assert (count, note, calls) == (10, "", [])


def test_probe_roots_are_kramers_even_and_clamped_to_the_space():
    assert probe_roots(2, n_elec=1, n_active=10, margin=8) == 10        # C(10,1) = 10
    assert probe_roots(2, n_elec=1, n_active=30, margin=7) == 10        # 9 -> 10, odd N
    assert probe_roots(6, n_elec=2, n_active=20, margin=3) == 9         # even N: no rounding



# --- the budget -------------------------------------------------------------------------------

def test_a_memory_budget_is_re_evaluated_at_the_filling_it_is_asked_about():
    """⚠ The conventional-CI ceiling bounds the **determinant count**, not the spinor count: a
    dilute space well past 22 spinors runs where a half-filled one of 24 refuses. A budget
    that compared spinor counts alone would refuse the first and admit the second."""
    budget = proto.resolve_budget(solver="ci", n_elec=1, n_roots=2, available_gb=8.0)
    assert budget.fits(30, 1)
    assert not budget.fits(30, 15)


def test_a_stated_determinant_budget_bounds_the_determinant_count():
    budget = proto.resolve_budget(solver="ci", n_elec=6, n_roots=5, max_determinants=1000)
    from kuiva.ci.strings import cas_dimension

    assert cas_dimension(budget.max_spinors, 6) <= 1000
    assert cas_dimension(budget.max_spinors + 2, 6) > 1000
    assert budget.max_spinors % 2 == 0                     # whole Kramers pairs


def test_two_statements_of_one_size_are_refused():
    with pytest.raises(ValueError, match="not both"):
        proto.resolve_budget(solver="ci", n_elec=2, n_roots=2, max_spinors=20,
                             max_determinants=500)


def test_the_network_budget_is_provisional_and_says_so():
    budget = proto.resolve_budget(solver="dmrg", n_elec=2, n_roots=4)
    assert budget.provisional and "provisional" in budget.describe()


def test_an_unconfigured_memory_limit_refuses_rather_than_guessing_a_size():
    with pytest.raises(ValueError, match="no memory limit is configured"):
        proto.resolve_budget(solver="ci", n_elec=2, n_roots=2, available_gb=float("inf"))


# --- pruning ------------------------------------------------------------------------------------

def _fake_probe(space, spectrum_cm, entropy, *, cpu=1.0) -> ProbeResult:
    """A :class:`ProbeResult` with the two numbers a decision reads, and nothing real behind.

    ``entropy`` is per **spinor** of the active space, in the order ``space.spaces.active``
    lists them, exactly as a real probe's is.
    """
    entropy = np.asarray(entropy, dtype=float)
    n = int(space.spaces.n_active)
    assert entropy.size == n
    return ProbeResult(space=space, coeff=np.zeros((2, n), dtype=complex), result=None,
                       spectrum_cm=np.asarray(spectrum_cm, dtype=float), entropy=entropy,
                       mutual_information=np.zeros((n, n)), occupations=np.full(n, 0.5),
                       n_roots=len(spectrum_cm), n_determinants=100, converged=True,
                       cpu_seconds=cpu)


def _set(cls, pairs, *, fixed=False, occupations=None) -> cand.CandidateSet:
    """A candidate set in the constructors' own convention.

    ⚠ ``occupations`` and ``ranking`` carry **one value per Kramers pair** while ``columns``
    carries two per pair -- an asymmetry every real constructor obeys and every mask over
    them has to.
    """
    columns = cand._pair_columns(pairs)
    n_pairs = columns.size // 2
    occ = (np.zeros(n_pairs) if occupations is None
           else np.asarray(occupations, dtype=float))
    assert occ.size == n_pairs
    return cand.CandidateSet(cls=cls, columns=columns, description="{} test set".format(cls),
                             occupations=occ, ranking=np.zeros(n_pairs), fixed=fixed)


def _space_over(columns, n_elec=2, n_orb=40):
    from kuiva.mcscf.casci import active_space

    return active_space(columns, n_orb, n_elec_total=n_elec + 10, n_active_elec=n_elec)


def test_pruning_keeps_whole_kramers_pairs():
    candidate = _set("bridge", [3, 4, 5])
    space = _space_over(candidate.columns)
    result = _fake_probe(space, [0.0, 1.0], [1.0, 1.0, 1.0, 1.0, 0.001, 0.001])
    kept, note = proto.prune_candidates(candidate, result, prune_rel=0.1)
    assert list(kept) == [6, 7, 8, 9]
    assert "pruned 1 of 3" in note


def test_pruning_rounds_a_tie_outward_so_a_degenerate_group_is_never_split():
    """⚠ Two candidates that are degenerate to the last bits sit either side of the cut.

    That is the *only* way a threshold can split a tie, and it is not hypothetical: two
    symmetry partners carry the same entropy up to rounding, so a cut that happens to land
    between them keeps one of a pair whose members are physically indistinguishable -- an
    active space that breaks a degeneracy the molecule has exactly. Here the cut is 0.5 and
    the two candidates are 0.5 +- 1e-9, which the project's own relative-gap tolerance calls
    degenerate; both are kept and the note says why.
    """
    candidate = _set("bridge", [3, 4, 5])
    space = _space_over(candidate.columns)
    result = _fake_probe(space, [0.0, 1.0],
                         [1.0, 1.0, 0.5 + 1e-9, 0.5 + 1e-9, 0.5 - 1e-9, 0.5 - 1e-9])
    kept, note = proto.prune_candidates(candidate, result, prune_rel=0.5)
    assert list(kept) == [6, 7, 8, 9, 10, 11]
    assert "split a tie" in note


def test_pruning_refuses_a_candidate_that_was_not_in_the_probed_space():
    candidate = _set("bridge", [3])
    space = _space_over(cand._pair_columns([7, 8]))
    result = _fake_probe(space, [0.0, 1.0], np.ones(4))
    with pytest.raises(ValueError, match="not in the probed active space"):
        proto.prune_candidates(candidate, result)


def test_exclude_refuses_to_cut_a_fixed_class():
    """A shell and a correlating shell are whole or nothing; ``exclude=`` inside one is a
    request to cut it, and the refusal names the statement."""
    shell = _set("shell", [3, 4, 5], fixed=True)
    with pytest.raises(ValueError, match="whole or not at all"):
        proto._drop_excluded(shell, [6, 7])


def test_exclude_narrows_an_ordinary_class_in_whole_pairs():
    bridge = _set("bridge", [3, 4, 5])
    narrowed = proto._drop_excluded(bridge, [8, 9])
    assert list(narrowed.columns) == [6, 7, 10, 11]
    assert any("exclude=" in n for n in narrowed.notes)


# --- the round loop, against a scripted probe -------------------------------------------------

@pytest.fixture(scope="module")
def ticl3():
    return reference_for("ticl3")


class _ScriptedProbe:
    """A stand-in for :func:`kuiva.autocas.probe.probe` with a scripted spectrum per size.

    The round loop's inputs are a spectrum and an entropy vector; scripting them is what makes
    the *decision* testable at all. ``shift_per_pair`` moves the second state of the target
    manifold by a fixed amount for every candidate pair in the space beyond the core, which is
    the shape of a real class that matters: it changes the splitting inside the ground
    manifold.
    """

    def __init__(self, n_core, *, shift_per_pair=0.0, entropy_of=None, fail_at=None):
        self.n_core = n_core
        self.shift_per_pair = shift_per_pair
        #: ``entropy_of(position)`` -- the **position** in the active space rather than the
        #: spinor index, so a test scripts "the lowest pair is uninteresting" without having
        #: to know which columns a real candidate construction chose.
        self.entropy_of = entropy_of or (lambda position: 1.0)
        self.fail_at = fail_at
        self.calls = []

    def __call__(self, reference, coeff, space, *, n_roots, budget=None, report=False,
                 seed=None):
        n_extra = (space.n_active - self.n_core) // 2
        self.calls.append(space.n_active)
        if self.fail_at is not None and space.n_active == self.fail_at:
            raise SolverFailure("scripted failure at {} spinors".format(space.n_active))
        split = 20.0 + self.shift_per_pair * n_extra
        spectrum = [0.0, split, 4000.0, 4000.0 + split]
        entropy = np.asarray([self.entropy_of(i)
                              for i in range(int(space.spaces.n_active))])
        return _fake_probe(space, spectrum, entropy, cpu=0.5)


def _assemble_ticl3(ticl3, monkeypatch, scripted, **kwargs):
    # The same script stands in for the fixed-orbital measurement every verdict is taken on
    # and for the one pre-optimization of the accepted space at the end.
    monkeypatch.setattr(proto, "measure", scripted)
    monkeypatch.setattr(proto, "probe", scripted)
    return proto.assemble(ticl3, report=False, **kwargs)


def test_a_class_that_moves_the_manifold_is_kept_and_one_that_does_not_is_dropped(
        ticl3, monkeypatch):
    """The decider, both ways round, on one real candidate construction.

    The same targets, the same candidates and the same pruning; only the scripted spectrum
    differs, and that alone decides the class. That is the property the protocol claims.
    """
    targets = ["shells", ("bonding", "Ti", 2)]
    moved = _assemble_ticl3(ticl3, monkeypatch, _ScriptedProbe(10, shift_per_pair=100.0),
                            targets=targets)
    assert [r.verdict for r in moved.rounds] == ["core", "kept"]
    assert moved.n_active > 10

    still = _assemble_ticl3(ticl3, monkeypatch, _ScriptedProbe(10, shift_per_pair=0.0),
                            targets=targets)
    assert [r.verdict for r in still.rounds] == ["core", "dropped"]
    assert still.n_active == 10


def test_a_change_under_the_tolerance_but_over_the_noise_is_kept_as_inconclusive(
        ticl3, monkeypatch, kuiva_caplog):
    """⚠ Three verdicts, not two. A larger space is the safe error where the probe cannot
    resolve the question, and the budget is what bounds the consequence."""
    assembly = _assemble_ticl3(
        ticl3, monkeypatch, _ScriptedProbe(10, shift_per_pair=5.0),
        targets=["shells", ("bonding", "Ti", 2)], spectrum_tol_cm=50.0, probe_noise_cm=1.0)
    assert assembly.rounds[1].verdict == "kept (inconclusive)"
    assert any("under the tolerance" in r.getMessage() and "KEPT" in r.getMessage()
               for r in kuiva_caplog.records)


def test_a_probe_that_failed_keeps_the_class_out_and_the_run_going(ticl3, monkeypatch):
    """⚠ An untested addition is not an accepted one, and a failure at a trial point is not a
    reason to lose a run that has already paid for its core."""
    scripted = _ScriptedProbe(10, shift_per_pair=100.0, fail_at=14)
    assembly = _assemble_ticl3(ticl3, monkeypatch, scripted,
                               targets=["shells", ("bonding", "Ti", 2)])
    assert assembly.rounds[1].verdict == "dropped (not measured)"
    assert assembly.n_active == 10


def test_a_class_over_budget_is_dropped_with_its_size_printed_and_never_probed(
        ticl3, monkeypatch):
    scripted = _ScriptedProbe(10, shift_per_pair=100.0)
    assembly = _assemble_ticl3(ticl3, monkeypatch, scripted,
                               targets=["shells", ("bonding", "Ti", 2)], max_spinors=10)
    assert assembly.rounds[1].verdict == "dropped (over budget)"
    assert "against a budget of 10" in assembly.rounds[1].note
    assert scripted.calls == [10, 10]      # the core's measurement and its final probe only


def test_the_core_alone_over_budget_refuses_before_the_first_probe(ticl3, monkeypatch):
    """⚠ A shell is never cut to fit, and the refusal names the two ways out."""
    scripted = _ScriptedProbe(10)
    monkeypatch.setattr(proto, "measure", scripted)
    monkeypatch.setattr(proto, "probe", scripted)
    with pytest.raises(ValueError, match="never cut"):
        proto.assemble(ticl3, targets="shells", max_spinors=8, report=False)
    assert scripted.calls == []


def test_a_pruned_class_is_re_probed_before_it_is_judged(ticl3, monkeypatch):
    """A class is judged on the space that would actually be added, not on the pool it was
    offered from -- so a prune that removed anything costs a second probe, and the rounds
    table's cpu column carries both."""
    # TiCl3's two bonding candidates are spinors 68-71, below the 3d shell at 72-81, so in
    # the sorted active space they are the first two pairs. Give the first one nothing.
    scripted = _ScriptedProbe(10, shift_per_pair=100.0,
                              entropy_of=lambda i: 0.001 if i < 2 else 1.0)
    assembly = _assemble_ticl3(ticl3, monkeypatch, scripted,
                               targets=["shells", ("bonding", "Ti", 2)])
    record = assembly.rounds[1]
    assert record.n_pruned > 0
    assert record.cpu_seconds == pytest.approx(1.0)    # two probes at 0.5 s each


def test_every_accepted_space_is_whole_kramers_pairs_and_carries_its_statement(
        ticl3, monkeypatch):
    assembly = _assemble_ticl3(ticl3, monkeypatch, _ScriptedProbe(10, shift_per_pair=100.0),
                               targets=["shells", ("bonding", "Ti", 2)])
    columns = np.sort(np.asarray(assembly.space.spaces.active, dtype=int))
    assert columns.size % 2 == 0
    assert np.array_equal(columns[1::2], columns[0::2] + 1)
    assert np.all(columns[0::2] % 2 == 0)
    assert "AVAS" in assembly.space.description
    assert "[" not in assembly.space.description       # a description carries no index list


def test_a_probe_budget_below_the_sites_hund_configurations_is_refused():
    """⚠ The mechanism behind the Tier-3 run whose every verdict was an artefact: three d^5
    sites need 32**3 = 32 768 determinants before a single charge-transfer one, and a
    selected CI capped at 6000 returned spectra that were not spectra. Refused before any
    measurement, with the number to state; a space whose whole CAS fits is never refused."""
    from types import SimpleNamespace

    centre = SimpleNamespace(l=2, atoms=(0, 1, 2))
    shell = SimpleNamespace(electrons=15.0)
    space = _space_over(np.arange(30), n_elec=15, n_orb=60)
    with pytest.raises(ValueError, match="32768 determinants"):
        proto._check_probe_budget(ProbeBudget(max_determinants=6000), space, [centre], shell)
    proto._check_probe_budget(ProbeBudget(max_determinants=40000), space, [centre], shell)
    small = _space_over(np.arange(10), n_elec=1)
    proto._check_probe_budget(ProbeBudget(max_determinants=10), small,
                              [SimpleNamespace(l=2, atoms=(0,))], SimpleNamespace(electrons=1.0))


def _centre_of(l, atoms, where):
    from kuiva.autocas import centres as ctr

    return ctr.Centre(atoms=tuple(atoms), l=l, labels=(where,), population=0.9, pooled=False,
                      reference_state="", reference_channels=(), stated=False)


def test_the_floors_of_centres_of_different_l_are_their_own_and_the_product_is_over_blocks():
    """⚠ The mechanism: a floor is computed from each centre's OWN electron count. Dividing
    the shell's total over the atoms -- right for equivalent centres -- turns a Cu(2+) d^9
    beside a Ce(3+) f^1 into two "d^5/f^5 atoms" and states a 6 x 6 manifold where the
    physics is a doublet times the 2F5/2 sextet."""
    from types import SimpleNamespace

    centres = [_centre_of(2, (0,), "1 Cu"), _centre_of(3, (1,), "2 Ce")]
    shell = SimpleNamespace(electrons=10.0, centre_electrons=(9.0, 1.0))
    floor, product, terms = proto._floors(centres, shell)
    assert (floor, product) == (6, 12), terms
    assert "2F5/2" in terms
    dimension, _ = proto._hund_product(centres, shell)
    assert dimension == 10 * 14
    averaged = SimpleNamespace(electrons=10.0)          # no per-centre counts: the old rule
    assert proto._floors(centres, averaged)[:2] != (6, 12)


def test_an_atom_with_two_shells_is_one_site_holding_both():
    """A 4f and a 5d on one cerium are two centres and ONE site; the site's spinor count is
    both shells, and a localization handed an equal split would put f orbitals on the
    titanium."""
    centres = [_centre_of(3, (0,), "1 Ce"), _centre_of(2, (0,), "1 Ce"),
               _centre_of(2, (1,), "2 Ti")]
    sites = proto._centre_atoms(centres)
    assert sites == ((0,), (1,))
    assert proto._site_counts(centres, sites) == (24, 10)
    assert proto._centre_atoms(centres[:2]) == ()


def test_targets_naming_shells_of_two_l_resolve_instead_of_refusing(monkeypatch):
    """The refusal is lifted where the targets are resolved; the composition itself is the
    candidate construction's (one projection onto the union)."""
    from types import SimpleNamespace

    from kuiva.autocas import centres as ctr

    made = {("Ti",): _centre_of(2, (0,), "1 Ti"), ("Ce",): _centre_of(3, (1,), "2 Ce")}
    monkeypatch.setattr(ctr, "shell_centre", lambda reference, atoms, l: made[tuple(atoms)])
    targets = [SimpleNamespace(detected=False, atoms=("Ce",), l="f"),
               SimpleNamespace(detected=False, atoms=("Ti",), l="d")]
    centres = proto._resolve_centres(None, targets, report=False)
    assert [c.l for c in centres] == [2, 3]


def test_embedded_determinants_carry_the_added_pairs_reference_occupations():
    """The nesting a trial measurement starts from: the accepted determinants over the larger
    space, a doubly occupied added pair filled, an empty one left empty, and a singly
    occupied one taken both ways -- so the embedded list has the trial's particle number."""
    from kuiva.autocas.probe import embed_determinants
    from kuiva.ci.strings import Determinants, popcount

    small = _space_over([4, 5, 6, 7], n_elec=1)
    dets = Determinants.from_occupations([[0], [1], [2], [3]], 4)
    occupation = np.zeros(40)
    occupation[[0, 1]] = 1.0           # a doubly occupied pair below the space
    occupation[[8, 9]] = 0.5           # a singly occupied pair
    trial = _space_over([0, 1, 4, 5, 6, 7, 8, 9, 10, 11], n_elec=4)
    seeded = embed_determinants(dets, small, trial, occupation)
    assert seeded.ndet == 2 * dets.ndet
    assert np.all(popcount(seeded.masks) == 4)
    assert np.all(seeded.masks & np.uint64(0b11) == np.uint64(0b11))     # the filled pair
    assert np.all(seeded.masks & np.uint64(0b11 << 8) == 0)               # the empty pair
    somo = (seeded.masks >> np.uint64(6)) & np.uint64(0b11)
    assert sorted(set(int(x) for x in somo)) == [1, 2]
    with pytest.raises(ValueError, match="do not embed"):
        embed_determinants(dets, small, _space_over([0, 1, 4, 5, 6, 7], n_elec=1),
                           occupation)


def test_a_seeded_cheap_ci_keeps_every_seed_determinant_and_adds_within_its_budget():
    """⚠ Nesting is what makes two measurements comparable: a selection started afresh in a
    larger space may drop what the smaller one held."""
    from kuiva.ci.strings import Determinants
    from kuiva.mcscf.preopt import cheap_ci

    rng = np.random.default_rng(7)
    n = 8
    h = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    h = h + h.conj().T + np.diag(np.arange(n, dtype=float) * 2.0)
    eri = np.zeros((n, n, n, n), dtype=complex)
    seed = Determinants.from_occupations([[0, 1, 2], [0, 1, 3], [0, 2, 5], [4, 6, 7]], n)
    ci = cheap_ci(h, eri, 3, n_states=2, max_determinants=5, seed=seed, with_2rdm=False)
    assert np.all(ci.dets.positions(seed.masks) >= 0)
    assert seed.ndet < ci.n_determinants <= seed.ndet + 5


def test_irrelevant_pairs_do_not_move_a_fixed_orbital_measurement_whatever_their_number(ticl3):
    """⚠ The mechanism the verdicts are taken at fixed orbitals for. When two pre-optimized
    probes were compared, four to seven Kramers pairs describing nothing moved the target
    spectrum by 58-102 cm^-1 -- above the tolerance -- because the optimizer rotated into
    them. At fixed orbitals, with the trial nested in the core, the highest-energy virtual
    pairs (unique: eigenvectors of the pair-folded inactive Fock) move it by far less than
    the noise floor at every size."""
    from kuiva.autocas.probe import measure
    from kuiva.mcscf.casci import active_space
    from kuiva.mcscf.orbopt import CASIntegrals
    from kuiva.spinor.expand import fold_to_kramers_pairs, rotate_kramers_pairs

    centres = ctr.detect_centres(ticl3, report=False).centres
    construction = cand.shell_candidates(ticl3, centres, report=False)
    space = proto._space_of(ticl3, [construction.shell])
    coeff = construction.coeff
    ints = CASIntegrals.build(ticl3.factors, ticl3.h_one_electron(), coeff, space.spaces,
                              e_nuc=ticl3.data.e_nuc)
    virtual = np.asarray(space.spaces.virtual, dtype=int)
    f_pair, _ = fold_to_kramers_pairs(ints.f_inactive, columns=virtual)
    _, v = np.linalg.eigh(f_pair)
    coeff = rotate_kramers_pairs(coeff, v, virtual)       # virtual columns by orbital energy
    budget = ProbeBudget()
    core = measure(ticl3, coeff, space, n_roots=10, budget=budget)
    assert core.fixed_orbitals and core.coeff is not None
    reference = rts.target_spectrum(core.spectrum_cm, 2)
    for n_pairs in (1, 4, 7):
        extra = virtual[-2 * n_pairs:]
        trial = active_space(np.sort(np.concatenate([space.spaces.active, extra])),
                             int(ticl3.nspinor), int(ticl3.data.nelec_total),
                             n_active_elec=space.n_elec)
        moved = measure(ticl3, coeff, trial, n_roots=10, budget=budget, seed=core)
        distance, _ = proto.spectrum_distance(reference,
                                              rts.target_spectrum(moved.spectrum_cm, 2))
        assert distance < 0.1 * proto.DEFAULT_PROBE_NOISE_CM, (n_pairs, distance)


def test_a_truncated_core_of_coupled_centres_is_not_measured():
    """⚠ Measured on mn3_linear: a selected CI of three coupled d^5 sites is not a spin
    manifold at 6000 or 40 000 determinants, and seeded with the Hund product it puts the
    ferromagnetic S = 15/2 lowest with its components split by more than the exchange. A
    single centre, or a complete core of several, is measured as before."""
    from types import SimpleNamespace

    budget = ProbeBudget(max_determinants=40000)
    trimer = [SimpleNamespace(l=2, atoms=(0, 1, 2))]
    big = _space_over(np.arange(30), n_elec=15, n_orb=60)
    assert "KEPT UNMEASURED" in proto._coupled_and_truncated(big, trimer, budget)
    dimer = [SimpleNamespace(l=2, atoms=(0, 1))]
    assert proto._coupled_and_truncated(_space_over(np.arange(20), n_elec=2), dimer,
                                        budget) == ""       # CAS(2, 20) is complete
    one = [SimpleNamespace(l=2, atoms=(0,))]
    assert proto._coupled_and_truncated(big, one, budget) == ""


def test_classes_on_an_unmeasurable_core_are_kept_without_a_measurement(ticl3, monkeypatch):
    scripted = _ScriptedProbe(10, shift_per_pair=0.0)
    monkeypatch.setattr(proto, "_coupled_and_truncated", lambda *a: "scripted reason")
    assembly = _assemble_ticl3(ticl3, monkeypatch, scripted,
                               targets=["shells", ("bonding", "Ti", 2)])
    assert [r.verdict for r in assembly.rounds] == ["core", "kept (not measurable)"]
    assert assembly.n_active > 10
    assert assembly.decided is None
    assert len(scripted.calls) == 1                # the final probe only
    assert "scripted reason" in assembly.notes


def test_a_target_set_without_a_shell_is_refused(ticl3):
    with pytest.raises(ValueError, match="needs at least one shell"):
        proto.assemble(ticl3, targets=[("frontier", "Cl")], report=False)


def test_a_class_naming_an_atom_that_is_not_a_centre_is_refused(ticl3):
    with pytest.raises(ValueError, match="not a centre"):
        proto.assemble(ticl3, targets=["shells", ("bonding", "Cl")], report=False)


def test_require_takes_a_character_statement(ticl3, monkeypatch):
    """``require=`` pins orbitals the probe cannot see."""
    assembly = _assemble_ticl3(ticl3, monkeypatch, _ScriptedProbe(14),
                               targets="shells", require=[("character", "Cl", "p", 4)])
    assert assembly.n_active == 14
    assert "Kramers pairs of l=1 character" in assembly.space.description


def test_require_takes_an_avas_statement_as_part_of_the_cores_projection(ticl3, monkeypatch):
    """An AVAS pin is one more entry of the core's union projection, with its own count, and
    carries its pairs' reference electrons into the space. It is refused where it would name
    a shell the core already projects onto, and ``exclude=`` does not take the form at all."""
    assembly = _assemble_ticl3(ticl3, monkeypatch, _ScriptedProbe(16),
                               targets="shells", require=[("avas", "Cl2", "p", 6)])
    assert (assembly.n_active, assembly.space.n_elec) == (16, 7)
    assert "attributed to the p reference shell of 2 Cl (require=" in assembly.space.description
    assert assembly.sets[0].n_pairs == 5, "the shell keeps its own five pairs"
    with pytest.raises(ValueError, match="more than one entry"):
        proto.assemble(ticl3, targets="shells", require=[("avas", "Ti", "d", 2)],
                       report=False)
    with pytest.raises(ValueError, match="require= form only"):
        proto.assemble(ticl3, targets="shells", exclude=[("avas", "Cl2", "p", 2)],
                       report=False)
    with pytest.raises(ValueError, match="n_spinors"):
        proto.assemble(ticl3, targets="shells", require=[("avas", "Cl2", "p")], report=False)


# --- Tier 1: the real probe -----------------------------------------------------------------

@pytest.mark.slow
def test_ticl3_at_level_zero_is_the_committed_space_and_a_clean_boundary(ticl3):
    """The default run on a system whose active space is externally validated.

    The space has to be the committed one (the Tier-2 reference is defined by it), and the
    proposal has to be a *boundary*: an even count -- this is an odd-electron system, so every
    level is at least two-fold -- at a gap the state-average diagnostic would call
    unambiguous.
    """
    system = sysdef.get("ticl3")
    assembly = proto.assemble(ticl3, targets="shells", report=False)
    assert assembly.n_active == 2 * system.ncas
    assert assembly.space.n_elec == system.nelecas
    assert assembly.proposal.n_states is not None
    assert assembly.proposal.n_states % 2 == 0
    assert assembly.proposal.boundary.found
    assert assembly.proposal.boundary.gap_cm > DEFAULT_MANIFOLD_GAP_CM
    assert assembly.proposal.n_states >= assembly.floor


@pytest.mark.slow
def test_tif3_at_level_zero_is_the_committed_space_and_a_clean_boundary():
    """The same claim on the second committed ``d^1`` system, whose ligand is far more
    electronegative: a shell construction that only reproduced TiCl3 would be one tuned to it."""
    system = sysdef.get("tif3")
    reference = reference_for("tif3")
    assembly = proto.assemble(reference, targets="shells", probe_budget=ProbeBudget(max_iter=4),
                              report=False)
    assert assembly.n_active == 2 * system.ncas
    assert assembly.space.n_elec == system.nelecas
    assert len(assembly.centres) == 1 and assembly.centres[0].where.endswith("Ti")
    assert assembly.proposal.boundary.found
    assert assembly.proposal.n_states is not None
    assert assembly.proposal.n_states % 2 == 0
    assert assembly.proposal.n_states >= assembly.floor
    assert assembly.proposal.boundary.gap_cm > DEFAULT_MANIFOLD_GAP_CM


@pytest.mark.slow
def test_fecl2_offers_its_bonding_combinations_and_decides_them_by_a_number():
    """A bonding round on the high-spin ``d^6`` shell: the class has candidates (the sigma
    combinations below the shell's cut), is probed, and its verdict carries the distance and
    the tolerance it was taken on -- whichever way it goes. ⚠ What is asserted is that the
    decision is *attributable*, not which way it falls: the verdict is a measurement, and it
    is recorded in the notes rather than pinned here."""
    reference = reference_for("fecl2")
    assembly = proto.assemble(reference, targets=["shells", ("bonding", "Fe")],
                              probe_budget=ProbeBudget(max_iter=4), report=False)
    shell_pairs = assembly.sets[0].n_pairs
    assert shell_pairs == 5
    bonding = [r for r in assembly.rounds if r.cls == "bonding"]
    assert len(bonding) == 1
    record = bonding[0]
    assert record.n_candidates >= 1
    assert record.verdict in ("kept", "kept (inconclusive)", "dropped",
                              "dropped (all pruned)"), record
    if record.verdict != "dropped (all pruned)":
        assert record.distance_cm is not None and record.tolerance_cm is not None
        assert record.tolerance_cm >= proto.DEFAULT_SPECTRUM_TOL_CM
    kept = [s for s in assembly.sets if s.cls == "bonding"]
    assert bool(kept) == record.kept
    extra = sum(s.n_pairs for s in kept)
    assert assembly.n_active == 2 * (shell_pairs + extra)
    assert np.all(np.asarray(assembly.space.spaces.active)[0::2] % 2 == 0)


# --- the stage, and the handoff ----------------------------------------------------------

def _scalar(key, **kwargs):
    """A finished ScalarSCF stage for a committed system."""
    import kuiva

    system = sysdef.get(key)
    molecule = kuiva.Molecule(atoms=system.atoms, basis=system.basis, charge=system.charge,
                              spin=system.spin)
    options = dict(screening="none", memory_gb=8.0, atomic_reference=True)
    options.update(kwargs)
    return kuiva.ScalarSCF(molecule, **options).run()


@pytest.fixture(scope="module")
def ticl3_reference_stage():
    import kuiva

    return kuiva.Reference(_scalar("ticl3")).run()


def test_the_stage_refuses_without_the_free_atom_reference_orbitals():
    """⚠ Eagerly, at construction: every shell it builds is a projection onto them, and a
    prerequisite that fails an hour into a run is a prerequisite nobody checked."""
    import kuiva

    reference = kuiva.Reference(_scalar("ticl3", atomic_reference=False)).run()
    with pytest.raises(ValueError, match="atomic_reference=True"):
        kuiva.AutoCAS(reference)


def test_the_stage_refuses_a_misspelled_target_and_two_statements_of_one_size(
        ticl3_reference_stage):
    import kuiva

    with pytest.raises(ValueError, match="not a target"):
        kuiva.AutoCAS(ticl3_reference_stage, targets="everything")
    with pytest.raises(ValueError, match="not both"):
        kuiva.AutoCAS(ticl3_reference_stage, max_spinors=20, max_determinants=500)
    with pytest.raises(TypeError, match="unexpected option"):
        kuiva.AutoCAS(ticl3_reference_stage, probe=dict(max_itr=4))


def test_an_unstated_state_count_is_still_one_state_without_an_autocas_upstream(
        ticl3_reference_stage):
    """⚠ The sentinel's whole purpose: "unstated" and "one state" stay different things, so a
    stage on an ordinary Reference keeps the default it always had."""
    import kuiva
    from kuiva.interface.stages import _UNSTATED, _states_from_upstream

    assert _states_from_upstream(ticl3_reference_stage, _UNSTATED, what="t") == (1, "")
    assert _states_from_upstream(ticl3_reference_stage, 7, what="t") == (7, "")
    cas = kuiva.CASSCF(ticl3_reference_stage, character=("Ti", "d"), n_active=10,
                       n_active_elec=1, max_iter=0, report=False)
    assert cas.n_states == 1 and cas.n_states_source == ""


@pytest.mark.slow
def test_the_stage_runs_level_zero_and_hands_the_proposal_on(ticl3_reference_stage):
    """The whole surface on the smallest system that has one: the space, the proposal, the
    site fallback, and the count the next stage takes without being told."""
    import kuiva
    from kuiva.interface.stages import _UNSTATED, _states_from_upstream

    auto = kuiva.AutoCAS(ticl3_reference_stage, probe=dict(max_iter=4), report=False).run()
    system = sysdef.get("ticl3")
    assert auto.space.n_active == 2 * system.ncas
    assert auto.space.n_elec == system.nelecas
    assert auto.n_states is not None and auto.n_states % 2 == 0
    assert auto.window.cutoff_cm > 0.0
    assert auto.sites is None                 # one centre: there is no partition to make
    assert list(auto.dmrg_ordering()) == list(auto.fiedler_ordering())
    count, source = _states_from_upstream(auto, _UNSTATED, what="CASSCF")
    assert count == auto.n_states and "AutoCAS" in source


@pytest.mark.slow
def test_the_handoff_adds_a_statement_and_not_a_calculation(ticl3_reference_stage):
    """⚠ **Bitwise**, not "close". The stage's claim is that inheriting the space, the
    orbitals and the count is the same calculation as spelling all three out -- if it were
    merely close, something in the handoff would be re-deriving a quantity rather than
    carrying it."""
    import kuiva

    auto = kuiva.AutoCAS(ticl3_reference_stage, probe=dict(max_iter=4), report=False).run()
    inherited = kuiva.CASSCF(auto, max_iter=3, report=False).run()
    # The same upstream, because the orbitals travel with the space: restating the selection
    # against the reference's own guess spinors would be a different calculation.
    explicit = kuiva.CASSCF(auto, active=list(auto.space.spaces.active),
                            n_active_elec=auto.space.n_elec, n_states=auto.n_states,
                            max_iter=3, report=False).run()
    assert float(inherited.energy) == float(explicit.energy)
    assert np.array_equal(np.asarray(inherited.energies), np.asarray(explicit.energies))

    # ⚠ And with no AutoCAS anywhere in the call: the layer under the stage API, handed the
    # three things the stage carries. The comparison above shares the upstream with the
    # inheriting run, so a quantity the stage re-derived from it would cancel there; here it
    # cannot, because the only inputs are the space, the orbitals and the count.
    from kuiva.interface import api

    plain = api.casscf(ticl3_reference_stage.reference, active=auto.space,
                       n_states=auto.n_states, coeff=auto.orbitals, max_iter=3, report=False)
    assert float(inherited.energy) == float(plain.energy)
    assert np.array_equal(np.asarray(inherited.energies), np.asarray(plain.ci.total_energies))


@pytest.mark.slow
def test_a_restated_character_is_refused_on_an_inherited_space(ticl3_reference_stage):
    """A selection is resolved against the orbitals it was stated on, and AVAS rotated
    these. ``active=`` -- a statement about the orbitals at hand -- is allowed."""
    import kuiva

    auto = kuiva.AutoCAS(ticl3_reference_stage, probe=dict(max_iter=4), report=False).run()
    for stage in (kuiva.CASSCF, kuiva.CASCI, kuiva.CheapCI):
        with pytest.raises(ValueError, match="character="):
            stage(auto, character=("Ti", "d"), n_active=10)
    kuiva.CASCI(auto, active=list(auto.space.spaces.active),
                n_active_elec=auto.space.n_elec, report=False)


@pytest.mark.slow
def test_a_site_blocked_graph_is_refused_without_a_site_partition(ticl3_reference_stage):
    """⚠ Refused rather than degraded to an entanglement order: a site-blocked ordering is a
    statement about which centre each orbital is on, and one centre makes no such statement."""
    import kuiva

    auto = kuiva.AutoCAS(ticl3_reference_stage, probe=dict(max_iter=4), report=False).run()
    with pytest.raises(ValueError, match="site-blocked"):
        kuiva.CASSCF(auto, solver="dmrg", graph="site-blocked",
                     solver_options=dict(max_bond=32))


@pytest.mark.slow
def test_cecl3_proposes_the_whole_ground_level_of_the_f_shell():
    """The f-block regime, where the floor is the LEVEL and not the spin multiplicity.

    Ce(3+) is ``4f^1``: the ground level is ``2F5/2``, six states, and the probe's own
    boundary is the gap to ``2F7/2``. ⚠ It is also the system whose 4f shell is *empty* in
    the scalar ROHF (the valence electron sits in a 5d orbital), so the electron count comes
    from the ion's reference state rather than from the orbitals, loudly.
    """
    import kuiva

    reference = kuiva.Reference(_scalar("cecl3")).run()
    auto = kuiva.AutoCAS(reference, probe=dict(max_iter=4), report=False).run()
    assert auto.space.n_active == 14 and auto.space.n_elec == 1
    assert auto.floor == 6
    assert auto.proposal.boundary.found and auto.n_states == 6
    assert auto.proposal.boundary.gap_cm > DEFAULT_MANIFOLD_GAP_CM


@pytest.mark.slow
def test_a_d_and_an_f_ion_assemble_into_one_space_whose_ground_manifold_is_their_product():
    """The protocol end to end on the committed d+f system. At the construction orbitals the
    lowest states have to be the 24-fold product of Ti 2D3/2 and Ce 2F5/2, split only by each
    ion's charge acting on the other (measured 5.4 cm^-1 over the lowest twenty) -- the check
    that one union projection handed the CI both ions' shells and not a mixture of them.

    ⚠ And the proposal is the whole manifold. The d-block floor is a spin doublet, so the
    floor-plus-margin count is 20 roots of the 24-fold manifold: averaged, the pre-optimization
    split it and the count read off that was 8. The probe now averages to the first
    fixed-orbital boundary past the ground manifold (60) and the proposal may not end inside a
    fixed-orbital manifold, which together give 24.
    """
    from kuiva.autocas.probe import ProbeBudget
    from test_autocas_centres import reference_for

    reference = reference_for("tice_far")
    assembly = proto.assemble(reference, targets=[("shell", "Ti", "d"), ("shell", "Ce", "f")],
                              probe_budget=ProbeBudget(max_iter=4), report=False)
    assert (assembly.space.n_elec, assembly.n_active) == (2, 24)
    assert (assembly.floor, assembly.product_floor) == (6, 12)
    assert assembly.site_atoms == ((0,), (1,)) and assembly.site_counts == (10, 14)
    decided = np.asarray(assembly.decided.spectrum_cm)
    assert decided.size == 20 and float(decided.max()) < 10.0, decided
    assert assembly.probe.n_roots == 60
    assert assembly.proposal.n_states == 24, assembly.proposal.describe()


@pytest.mark.slow
def test_the_spectrum_and_the_entropy_disagree_about_the_bridge_of_ti2cl6():
    """⚠ **The guard that cannot fail is the same defect as one that never fires.**

    Ti2Cl6 is two ``d^1`` titaniums bridged by two chlorines, and the bridging-ligand pair
    that mixes with the metal shell is exactly the superexchange pathway a published
    entanglement criterion would throw away: a bridging orbital shares at most ``ln 2`` nats
    with either metal, so in *absolute* terms it sits far below every metal orbital. This
    test asserts that the two rules reach **different** answers on a real system -- the
    spectrum keeps the class, an absolute entanglement cut at the same relative threshold
    would drop it -- because a keep/drop rule whose two candidate implementations never
    disagree is not a rule anybody chose.
    """
    import kuiva

    stage = kuiva.Reference(_scalar("ti2cl6")).run()
    # ⚠ The container, not the stage: `assemble` is the layer under the stage API.
    assembly = proto.assemble(stage.reference,
                              targets=["shells", ("bridge", ("Ti1", "Ti2"))],
                              probe_budget=ProbeBudget(max_iter=4), report=False)
    bridge_rounds = [r for r in assembly.rounds if r.cls == "bridge"]
    assert len(bridge_rounds) == 1
    assert bridge_rounds[0].kept, bridge_rounds[0]

    # What an absolute entanglement criterion would have said, at the same relative cut the
    # protocol uses *within* a class: the bridge pair against the largest entropy anywhere in
    # the space, which is a metal d orbital.
    bridge = [s for s in assembly.sets if s.cls == "bridge"][0]
    active = np.asarray(assembly.space.spaces.active, dtype=int)
    position = {int(c) // 2: i for i, c in enumerate(active[0::2])}
    s1 = assembly.probe.pair_entropy()
    rows = [position[int(c) // 2] for c in np.asarray(bridge.columns, dtype=int)[0::2]]
    assert float(np.max(s1[rows])) < proto.DEFAULT_PRUNE_REL * float(np.max(s1)), (
        "the entanglement criterion would have kept the bridge too, so this test no longer "
        "shows the two rules disagreeing: entropies {}".format(s1.tolist()))
