"""Stages 3-4: sampled-subspace CI, and a CASSCF driven by one .

What each tier here can fail on
--------------------------------
* **Recovery** (:mod:`kuiva.qc.recovery`) — the classical step between a shot record and a
  determinant space. Two of its tests are *mechanism* tests for defects that were live during
  development and that no energy would have exposed: a prior space given count *immunity*
  freezes the subspace at the first draw whenever a cap binds, and a fixed per-draw seed makes
  every proposal reproduce the incumbent. Both produce perfectly plausible converged
  calculations in which the sampler runs and changes nothing (the plausible-but-wrong shape).
* **The subspace solve** — against ``ci/strings.hamiltonian_matrix``, exactly as
  ``test_qc_mapping.py`` checks the mapping. ⚠ When the recovered space covers the whole CAS
  space this solver **is** a full CI, so the agreement proves the *pipeline* and nothing about
  the algorithm. :meth:`SQDSolver.coverage` is asserted to be 1.0 wherever that is the claim,
  so the distinction cannot be lost.
* **The adaptive-solver contract** — that ``solve`` does not sample, by **sample count**; that a
  proposal changes nothing until adopted; that the chart key is the determinant list.
* **Stage 4** — the whole loop: ``SQDSolver`` under the unchanged
  :func:`kuiva.mcscf.events.optimize_orbitals_events`, against the exact-CI CASSCF.

No framework anywhere: the stub backend is Kuiva's own exact simulator, so all of this runs in
the default laptop suite.
"""
import numpy as np
import pytest

from kuiva.ci.strings import (CASSpace, Determinants, determinant_memory_gb,
                              hamiltonian_matrix, kramers_partner)
from kuiva.integrals.transform import ThreeIndexAO
from kuiva.mcscf.adaptive import AdaptiveCISolver, SolverFailure
from kuiva.mcscf.casci import FullCISolver
from kuiva.mcscf.events import optimize_orbitals_events
from kuiva.mcscf.orbopt import CASIntegrals, OrbitalSpaces, optimize_orbitals
from kuiva.qc.algorithms import (algorithm_spec, available_algorithms, build_ci_solver,
                                 register_algorithm, unregister_algorithm)
from kuiva.qc.ansatz import (HardwareEfficientStrategy, ReferenceExcitationStrategy,
                             SamplingPlan, aufbau_mask, available_strategies,
                             resolve_strategy)
from kuiva.qc.backend import CapabilityError
from kuiva.qc.backends.stub import StubBackend
from kuiva.qc.recovery import POLICIES, recover_configurations, subspace_fraction
from kuiva.qc.sqd import SQDSolver, subspace_gb
from test_ci_strings import random_spinor_integrals

#: Machine precision: a sampled subspace that happens to be complete solves the *same* matrix
#: the reference does, so anything above rounding is a defect.
EXACT = 1e-11


class _Integrals:
    """The two methods ``CASIntegrals`` duck-typing needs, and nothing else.

    Deliberately not a ``CASIntegrals``: the solver is documented as duck-typed on
    ``h_active_effective()``/``active_eri()`` (as ``ttno_from_cas_integrals`` is), and a test
    that only ever passed the real class would not check that.
    """

    def __init__(self, h, eri, e_core=0.0):
        self._h, self._eri, self.e_core = h, eri, float(e_core)

    def h_active_effective(self):
        return self._h

    def active_eri(self):
        return self._eri


def _exact_spectrum(n, k, h, eri):
    return np.linalg.eigvalsh(hamiltonian_matrix(CASSpace(n, k).determinants(),
                                                 h, eri).toarray())


def _covering_solver(n_elec, **kwargs):
    """An SQD solver whose sampling covers the whole CAS space at these sizes."""
    kwargs.setdefault("shots", 8000)
    kwargs.setdefault("seed", 7)
    kwargs.setdefault("strategy", ReferenceExcitationStrategy(n_elec, theta=1.2))
    return SQDSolver(n_elec, **kwargs)


# --- configuration recovery ------------------------------------------------------------------

def test_projection_keeps_only_configurations_with_the_right_particle_number():
    masks = np.array([0b0011, 0b0111, 0b1010, 0b0000], dtype=np.uint64)
    counts = np.array([10, 5, 20, 3], dtype=np.int64)
    got = recover_configurations(masks, counts, n_elec=2, n_qubit=4, policy="project")
    assert got.determinants.masks.tolist() == [0b0011, 0b1010]
    assert got.counts.tolist() == [10, 20]
    assert got.shots == 38
    assert got.repaired == 0
    assert got.yield_fraction == pytest.approx(30 / 38)


def test_repair_moves_wrong_n_bitstrings_onto_the_particle_number():
    """Every shot ends up in the space, and every mask has exactly ``n_elec`` electrons."""
    masks = np.array([0b0011, 0b0111, 0b0001], dtype=np.uint64)
    counts = np.array([50, 5, 5], dtype=np.int64)
    got = recover_configurations(masks, counts, n_elec=2, n_qubit=4, policy="repair")
    assert np.all(np.array([bin(int(m)).count("1") for m in got.determinants.masks]) == 2)
    assert int(got.counts.sum()) == 60
    assert got.used_fraction == pytest.approx(1.0)
    # ⚠ The yield is the PRE-repair number: it describes the circuit, not the heuristic.
    assert got.yield_fraction == pytest.approx(50 / 60)


def test_a_prior_accumulates_counts_and_does_not_confer_immunity():
    """⚠ Mechanism test for a live defect: an immune prior freezes the space forever.

    Giving prior configurations a count above every sampled one makes them unevictable, so
    under a binding cap the recovered space is the prior space at every later draw — the
    sampler runs, spends shots, and can never change anything, while the calculation converges
    and looks entirely normal. Accumulation is the fix: prior counts are *added*, so a
    configuration seen once in the past loses to one seen often now.
    """
    prior_masks = np.array([0b0011, 0b0101], dtype=np.uint64)
    prior_counts = np.array([1, 1], dtype=np.int64)
    masks = np.array([0b1100, 0b1010, 0b0011], dtype=np.uint64)
    counts = np.array([100, 90, 2], dtype=np.int64)

    got = recover_configurations(masks, counts, n_elec=2, n_qubit=4, policy="project",
                                 max_determinants=2, prior=(prior_masks, prior_counts))
    # The two heavily sampled configurations win; the weakly supported prior ones are evicted.
    assert got.determinants.masks.tolist() == [0b1010, 0b1100]

    # And the accumulation is real: the prior count of 0b0011 adds to its new one.
    full = recover_configurations(masks, counts, n_elec=2, n_qubit=4, policy="project",
                                  prior=(prior_masks, prior_counts))
    tally = dict(zip(full.determinants.masks.tolist(), full.counts.tolist()))
    assert tally[0b0011] == 3                          # 2 sampled + 1 prior
    assert tally[0b0101] == 1                          # prior only, still present


def test_a_prior_with_the_wrong_particle_number_is_refused():
    with pytest.raises(ValueError, match="wrong particle number"):
        recover_configurations(np.array([0b0011], np.uint64), np.array([1], np.int64),
                               n_elec=2, n_qubit=4,
                               prior=(np.array([0b0111], np.uint64),
                                      np.array([1], np.int64)))


def test_an_unknown_recovery_policy_is_refused():
    """A name that resolved to a *different* procedure would be a research claim about
    something that never ran, so only implemented policies exist."""
    assert set(POLICIES) == {"project", "repair", "kramers"}
    with pytest.raises(ValueError, match="unknown recovery policy"):
        recover_configurations(np.array([0b11], np.uint64), np.array([1], np.int64),
                               n_elec=2, n_qubit=4, policy="parity")


def test_subspace_fraction_says_when_a_result_is_really_a_full_ci():
    assert subspace_fraction(20, 6, 3) == pytest.approx(1.0)
    assert subspace_fraction(10, 6, 3) == pytest.approx(0.5)


# --- circuit strategies -----------------------------------------------------------------------

def test_the_reference_strategy_concentrates_on_the_reference():
    """Stage A's whole claim: a hierarchy of excitations, not a uniform draw."""
    n, k = 6, 3
    circuit = ReferenceExcitationStrategy(k, theta=0.4).plan(n).circuits[0]
    probabilities = np.abs(StubBackend().statevector(circuit)) ** 2
    assert int(np.argmax(probabilities)) == aufbau_mask(k)
    assert probabilities[aufbau_mask(k)] > 0.5


def test_the_cx_ladder_costs_particle_number():
    """⚠ The documented reason ``entangle`` is off by default in Stage A.

    ``cx`` does not conserve particle number, so a ladder applied to the reference moves it
    off ``n_elec`` outright and the yield of valid configurations collapses. It is the package's
    point in miniature — a circuit family that assumes nothing about the Hamiltonian preserves
    none of its symmetries — and it is Stage B's cost, stated rather than discovered.
    """
    n, k = 8, 4
    backend = StubBackend()
    yields = []
    for entangle in (False, True):
        plan = ReferenceExcitationStrategy(k, theta=0.5, entangle=entangle).plan(n)
        sampled = backend.sample(plan.circuits[0], 4000, seed=1)
        got = recover_configurations(sampled.masks, sampled.counts, k, n, policy="project")
        yields.append(got.yield_fraction)
    assert yields[0] > 4 * yields[1], yields


def test_the_hardware_efficient_strategy_is_reproducible_from_its_seed():
    """⚠ A strategy that re-drew its angles per call would make ``solve`` non-deterministic,
    which the deterministic-solve contract forbids outright."""
    a = HardwareEfficientStrategy(3, layers=2, seed=5).plan(6).circuits[0]
    b = HardwareEfficientStrategy(3, layers=2, seed=5).plan(6).circuits[0]
    assert a == b
    assert HardwareEfficientStrategy(3, layers=2, seed=6).plan(6).circuits[0] != a


@pytest.mark.parametrize("shots,weights", [(1000, (1.0,)), (1000, (0.7, 0.3)),
                                           (7, (1.0, 1.0, 1.0)), (101, (0.5, 0.25, 0.25))])
def test_the_shot_allocation_is_exact_and_starves_nothing(shots, weights):
    """A circuit in a plan that is never measured is a strategy decision the strategy did not
    make, so every circuit gets at least one shot and the total is exact."""
    circuits = tuple(ReferenceExcitationStrategy(2, theta=0.1 * (i + 1)).plan(4).circuits[0]
                     for i in range(len(weights)))
    allocation = SamplingPlan(circuits, weights).allocate(shots)
    assert sum(allocation) == shots
    assert min(allocation) >= 1


def test_only_implemented_strategies_are_registered():
    assert set(available_strategies()) == {"reference", "hardware_efficient", "ucc",
                                           "cluster_jastrow", "time_evolution"}
    with pytest.raises(ValueError, match="unknown circuit strategy"):
        resolve_strategy("adapt_vqe", n_elec=2)


# --- Stage 5: Kramers-aware recovery ------------------------------------------

def _kramers_paired_integrals(n_spatial, seed):
    """Spin-free integrals expanded to interleaved Kramers-paired spinors.

    ⚠ Why this construction rather than random spinor integrals: the claim under test is about
    a **time-reversal-symmetric** Hamiltonian, and random complex integrals are not one. With
    SOC off and a Kramers-paired spinor set the symmetry is exact by construction, so a split
    Kramers pair can only come from the *subspace*, which is the thing being measured.
    """
    rng = np.random.default_rng(seed)
    h_sf = rng.normal(size=(n_spatial, n_spatial))
    h_sf = h_sf + h_sf.T
    e_sf = rng.normal(size=(n_spatial,) * 4)
    e_sf = e_sf + e_sf.transpose(1, 0, 2, 3)
    e_sf = e_sf + e_sf.transpose(0, 1, 3, 2)
    e_sf = 0.1 * (e_sf + e_sf.transpose(2, 3, 0, 1))
    n = 2 * n_spatial
    h = np.zeros((n, n), dtype=np.complex128)
    eri = np.zeros((n,) * 4, dtype=np.complex128)
    for s in (0, 1):
        h[s::2, s::2] = h_sf
        for t in (0, 1):
            eri[s::2, s::2, t::2, t::2] = e_sf
    return h, eri


def _is_time_reversal_closed(masks) -> bool:
    return set(masks.tolist()) == set(kramers_partner(masks).tolist())


def test_kramers_recovery_closes_the_space_under_time_reversal():
    masks = np.array([0b000011, 0b001001, 0b010100], dtype=np.uint64)
    counts = np.array([30, 20, 10], dtype=np.int64)
    got = recover_configurations(masks, counts, n_elec=2, n_qubit=6, policy="kramers")
    assert got.kramers_closed
    assert _is_time_reversal_closed(got.determinants.masks)
    assert got.closure_added > 0


def test_kramers_recovery_pools_the_counts_of_a_partner_pair():
    """Two partners are two measurements of one number, so they come out carrying its total.

    ⚠ ``0b0101`` occupies the *unbarred* spinor of each pair and ``0b1010`` the barred one, so
    they are partners. ``0b0011`` — both spinors of pair 0 — is its own partner, which is why
    the pooling has to be written to add rather than to double.
    """
    masks = np.array([0b0101, 0b1010, 0b0011], dtype=np.uint64)
    counts = np.array([30, 10, 7], dtype=np.int64)
    got = recover_configurations(masks, counts, n_elec=2, n_qubit=4, policy="kramers")
    assert got.determinants.masks.tolist() == [0b0011, 0b0101, 0b1010]
    assert got.counts.tolist() == [7, 40, 40]
    assert got.closure_added == 0


def test_a_kramers_cap_truncates_whole_orbits_and_is_met_from_below():
    """⚠ the whole-degenerate-group truncation rule at a third place in the code: half an orbit
    breaks the closure the policy exists to establish."""
    rng = np.random.default_rng(0)
    masks = np.unique(rng.choice(1 << 6, size=40, replace=False).astype(np.uint64))
    counts = rng.integers(1, 50, size=masks.size)
    for cap in (4, 7, 10):
        got = recover_configurations(masks, counts, n_elec=3, n_qubit=6, policy="kramers",
                                     max_determinants=cap)
        assert got.ndet <= cap
        assert _is_time_reversal_closed(got.determinants.masks)


def test_kramers_recovery_refuses_an_odd_mode_count():
    with pytest.raises(ValueError, match="Kramers-paired"):
        recover_configurations(np.array([0b011], np.uint64), np.array([1], np.int64),
                               n_elec=2, n_qubit=3, policy="kramers")


def test_a_time_reversal_closed_subspace_restores_the_kramers_degeneracy_exactly():
    """⚠ The claim the whole policy exists for, and it is a theorem rather than an average.

    A ``T``-symmetric Hamiltonian has exactly degenerate Kramers pairs. An arbitrary sampled
    subspace is not ``T``-invariant, so its projected Hamiltonian is not either and the pair
    splits by an arbitrary amount — landing squarely in the 1e-8..1e-6 Eh band reserved
    for a *genuine* numerical splitting, where it is indistinguishable from one by inspection.
    Closing the space puts the degeneracy back at machine precision.
    """
    n_spatial, n_elec = 4, 3
    n = 2 * n_spatial
    h, eri = _kramers_paired_integrals(n_spatial, seed=2)
    full = np.linalg.eigvalsh(hamiltonian_matrix(CASSpace(n, n_elec).determinants(),
                                                 h, eri).toarray())
    assert full[1] - full[0] < 1e-12, "the reference Hamiltonian must be T-symmetric"

    rng = np.random.default_rng(5)
    sampled = np.unique(rng.choice(CASSpace(n, n_elec).determinants().masks, size=14,
                                   replace=False))
    counts = rng.integers(1, 50, size=sampled.size)
    splittings = {}
    for policy in ("repair", "kramers"):
        got = recover_configurations(sampled, counts, n_elec=n_elec, n_qubit=n, policy=policy)
        spectrum = np.linalg.eigvalsh(
            hamiltonian_matrix(got.determinants, h, eri).toarray())
        splittings[policy] = float(spectrum[1] - spectrum[0])
    assert splittings["kramers"] < 1e-12
    assert splittings["repair"] > 1e-4


def test_a_kramers_sqd_solver_still_reproduces_the_exact_ci_at_full_coverage():
    """Closure adds determinants; it must not change the answer where the space was complete."""
    n, n_elec = 6, 3
    h, eri = _kramers_paired_integrals(3, seed=7)
    solver = _covering_solver(n_elec, recovery="kramers")
    energy, _gamma, _gamma2 = solver.solve(_Integrals(h, eri))
    assert solver.coverage() == pytest.approx(1.0)
    assert energy == pytest.approx(_exact_spectrum(n, n_elec, h, eri)[0], abs=EXACT)
    assert _is_time_reversal_closed(solver.last_recovery.determinants.masks)


# --- the subspace solve ------------------------------------------------------------------------

@pytest.mark.parametrize("n,k", [(6, 3), (8, 3), (8, 4)])
def test_full_coverage_reproduces_the_exact_ci(n, k):
    """Stage 3's exit criterion — and ⚠ read the coverage assertion before the energy one.

    When the recovered space covers the whole CAS space this solver **is** a full CI, so what
    this proves is the pipeline: mapping-free sampling, recovery, the subspace Hamiltonian,
    the RDMs and the state-averaging gate. It is not evidence that a realistic circuit concentrates its
    shots usefully, which is Stage 5's question and needs a Stage-C ansatz.
    """
    h, eri = random_spinor_integrals(n, seed=10 * n + k)
    solver = _covering_solver(k)
    energy, gamma, gamma2 = solver.solve(_Integrals(h, eri))
    assert solver.coverage() == pytest.approx(1.0), "this test's claim needs full coverage"
    assert abs(energy - _exact_spectrum(n, k, h, eri)[0]) < EXACT


def test_a_larger_cap_never_raises_the_energy():
    """The honest statement below full coverage — and ⚠ it is a statement about **nested**
    spaces, which is why the ladder is over ``max_determinants`` at a *fixed draw*.

    A cap truncates one sorted tally, so cap 5 ⊂ cap 10 ⊂ ... and the energies must fall. A
    ladder over *shot counts* would not be nested (see the next test) and asserting
    monotonicity there would be asserting a property of a random draw.
    """
    n, k = 8, 4
    h, eri = random_spinor_integrals(n, seed=99)
    exact = _exact_spectrum(n, k, h, eri)[0]
    spaces, energies = [], []
    for cap in (5, 10, 25, 70):
        solver = _covering_solver(k, max_determinants=cap)
        energies.append(solver.solve(_Integrals(h, eri))[0])
        spaces.append(set(solver._dets.masks.tolist()))
        assert energies[-1] > exact - EXACT, cap
    for smaller, larger in zip(spaces, spaces[1:]):
        assert smaller <= larger, "the cap ladder is not nested; monotonicity is not implied"
    assert energies == sorted(energies, reverse=True), energies
    assert abs(energies[-1] - exact) < EXACT


def test_more_shots_are_not_nested_so_the_energy_is_not_guaranteed_monotone():
    """⚠ Pins the *negative* result, because the positive one is the tempting claim.

    Two independent draws at different shot counts are independent samples, not a nested
    sequence: the larger draw can miss a configuration the smaller one found. Every shot
    count still gives a variational upper bound — that part is a theorem — but a shot ladder
    that happens to fall is evidence about a seed, and an example or a test that asserted it
    would be a noise detector.
    """
    n, k = 8, 4
    h, eri = random_spinor_integrals(n, seed=77)
    exact = _exact_spectrum(n, k, h, eri)[0]
    spaces = []
    for shots in (400, 1500, 6000):
        solver = SQDSolver(k, shots=shots, seed=3,
                           strategy=ReferenceExcitationStrategy(k, theta=1.0))
        assert solver.solve(_Integrals(h, eri))[0] > exact - EXACT, shots
        spaces.append(set(solver._dets.masks.tolist()))
    assert not all(a <= b for a, b in zip(spaces, spaces[1:])), (
        "these draws happened to nest; the test needs a seed where they do not, since the "
        "point is that nesting is not guaranteed")


def test_accumulating_proposals_never_raise_the_energy():
    """The other nested sequence, and the one SQD actually converges through: accumulation
    unions each fresh draw into the incumbent, so the space grows and the energy falls."""
    n, k = 8, 4
    h, eri = random_spinor_integrals(n, seed=123)
    ints = _Integrals(h, eri)
    solver = SQDSolver(k, shots=250, seed=2, accumulate=True,
                       strategy=ReferenceExcitationStrategy(k, theta=1.0))
    energy = solver.solve(ints)[0]
    space = set(solver._dets.masks.tolist())
    for _ in range(5):
        proposal = solver.propose(ints)
        if proposal is None:
            continue
        assert proposal.energy <= energy + EXACT
        solver.adopt(proposal.key)
        grown = set(solver._dets.masks.tolist())
        assert space <= grown, "accumulation must not lose a configuration"
        energy, space = proposal.energy, grown
    assert energy > _exact_spectrum(n, k, h, eri)[0] - EXACT


def test_the_rdms_satisfy_the_trace_conditions():
    """Cheap, strong, and sensitive to the phase bookkeeping the whole subspace solve rests
    on: ``tr gamma = N`` and ``sum_r Gamma_pqrr = (N-1) gamma_pq``."""
    n, k = 8, 4
    h, eri = random_spinor_integrals(n, seed=55)
    _, gamma, gamma2 = _covering_solver(k).solve(_Integrals(h, eri))
    assert abs(np.trace(gamma).real - k) < EXACT
    assert np.abs(gamma - gamma.conj().T).max() < EXACT
    assert np.abs(np.einsum("pqrr->pq", gamma2) - (k - 1) * gamma).max() < EXACT


# --- the adaptive-solver contract ------------------------------------------------------------------------

def test_the_solver_implements_the_protocol():
    assert isinstance(SQDSolver(2), AdaptiveCISolver)


def test_solve_samples_once_and_is_deterministic_thereafter():
    """⚠ The promise everything in :mod:`kuiva.mcscf.events` rests on, asserted by **sample
    count** rather than by comparing two energies.

    A ``solve`` that re-sampled would make ``E(kappa)`` discontinuous, and the trust region,
    the quadratic model and the accept/reject test would all be reading noise — silently. The
    first call must sample (there has to be an incumbent space); no later call may.
    """
    n, k = 6, 3
    h, eri = random_spinor_integrals(n, seed=21)
    solver = _covering_solver(k)
    first = solver.solve(_Integrals(h, eri))[0]
    assert solver.n_samples == 1
    for _ in range(3):
        assert solver.solve(_Integrals(h, eri))[0] == first
    assert solver.n_samples == 1

    # Smooth in the integrals at fixed space, which is the other half of the promise.
    perturbed = solver.solve(_Integrals(h + 1e-6 * np.eye(n), eri))[0]
    assert solver.n_samples == 1
    assert abs(perturbed - first) < 1e-4


def test_the_draw_seed_advances_so_a_proposal_can_differ():
    """⚠ Mechanism test for the second live defect.

    With one fixed per-call seed the backend replays an identical shot record; since the
    Stage-A/B circuits do not depend on the integrals either, every proposal then reproduces
    the incumbent space exactly, returns ``None``, and the event machinery spends shots to
    learn nothing. Measured as 0 adoptions out of every proposal before the fix.
    """
    solver = SQDSolver(3, shots=300, seed=4,
                       strategy=ReferenceExcitationStrategy(3, theta=1.0))
    assert solver.draw_seed() == 4
    n, k = 8, 3
    h, eri = random_spinor_integrals(n, seed=31)
    solver.solve(_Integrals(h, eri))
    assert solver.draw_seed() == 5, "a second draw would replay the first"

    # And the space really can grow: an accumulating proposal at a low shot count adds
    # configurations the first draw missed.
    before = solver.n_determinants
    proposal = solver.propose(_Integrals(h, eri))
    assert proposal is not None
    solver.adopt(proposal.key)
    assert solver.n_determinants > before


def test_propose_changes_nothing_until_it_is_adopted():
    n, k = 8, 3
    h, eri = random_spinor_integrals(n, seed=41)
    ints = _Integrals(h, eri)
    solver = SQDSolver(k, shots=400, seed=8,
                       strategy=ReferenceExcitationStrategy(k, theta=1.0))
    incumbent_energy = solver.solve(ints)[0]
    incumbent_key = solver.space_key()

    proposal = solver.propose(ints)
    assert proposal is not None
    assert solver.space_key() == incumbent_key
    assert solver.solve(ints)[0] == incumbent_energy

    # Accumulation makes the candidate a superset, so it is variationally at least as good.
    assert proposal.energy <= incumbent_energy + EXACT
    solver.adopt(proposal.key)
    assert solver.space_key() == proposal.key
    assert solver.solve(ints)[0] == pytest.approx(proposal.energy, abs=EXACT)


def test_adopting_a_stale_key_is_refused():
    n, k = 6, 3
    h, eri = random_spinor_integrals(n, seed=61)
    solver = _covering_solver(k)
    solver.solve(_Integrals(h, eri))
    with pytest.raises(ValueError, match="no proposal with key"):
        solver.adopt("not-a-key")


def test_the_space_key_is_the_determinant_list_and_nothing_finer():
    """⚠ Keying on the shot counts or the seed would declare a chart change at every proposal
    and clear the optimizer's curvature memory for nothing — the same rule
    ``DMRGSolver`` follows in keying on its caps rather than on observed bond dimensions."""
    from kuiva.mcscf.adaptive import array_key

    n, k = 6, 3
    h, eri = random_spinor_integrals(n, seed=71)
    solver = _covering_solver(k)
    assert solver.space_key() is None
    solver.solve(_Integrals(h, eri))
    assert solver.space_key() == array_key(CASSpace(n, k).determinants().masks)


def test_a_proposal_that_reproduces_the_incumbent_returns_none():
    n, k = 6, 3
    h, eri = random_spinor_integrals(n, seed=81)
    ints = _Integrals(h, eri)
    solver = _covering_solver(k)           # covers the whole space on the first draw
    solver.solve(ints)
    assert solver.propose(ints) is None


# --- refusals -------------------------------------------------------------------------------

def test_an_odd_state_count_on_an_odd_electron_system_is_refused():
    """State averaging through the same gate every other solver in Kuiva goes through — Kramers' theorem
    makes an odd count a split pair, and nothing downstream can recover it."""
    n, k = 6, 3
    h, eri = random_spinor_integrals(n, seed=91)
    solver = _covering_solver(k, n_states=3)
    with pytest.raises(ValueError, match="Kramers-degenerate block"):
        solver.solve(_Integrals(h, eri))


def test_asking_for_more_states_than_the_subspace_holds_is_a_solver_failure():
    """⚠ A :class:`SolverFailure`, not a crash: the event-gated optimizer turns it into a
    rejected step rather than a dead calculation."""
    n, k = 8, 4
    h, eri = random_spinor_integrals(n, seed=13)
    solver = SQDSolver(k, n_states=4, shots=200, max_determinants=2, seed=1,
                       strategy=ReferenceExcitationStrategy(k, theta=0.2))
    with pytest.raises(SolverFailure, match="recovered subspace"):
        solver.solve(_Integrals(h, eri))


def test_an_incapable_backend_is_refused_at_construction():
    """⚠ At construction, naming both sides — never mid-calculation, never emulated."""
    class _StatevectorOnly:
        name = "statevector_only"
        version = "0"

        def capabilities(self):
            return frozenset(("statevector",))

    with pytest.raises(CapabilityError) as excinfo:
        SQDSolver(2, backend=_StatevectorOnly())
    for expected in ("sqd", "sample", "statevector_only"):
        assert expected in str(excinfo.value)


# --- the algorithm registry (the second axis, refuse-never-reconcile) ------------------------------

def test_only_implemented_algorithms_are_registered():
    assert set(available_algorithms()) == {"sqd", "skqd", "vqe"}
    with pytest.raises(ValueError, match="unknown quantum CI algorithm"):
        algorithm_spec("qpe")


def test_build_ci_solver_refuses_the_pairing_before_constructing_anything():
    built = []

    def _factory(**kwargs):
        built.append(kwargs)
        return object()

    register_algorithm("test_only_algorithm", _factory, ("estimate",))
    try:
        with pytest.raises(CapabilityError, match="estimate"):
            build_ci_solver("test_only_algorithm", backend=_SamplerOnly())
        assert built == [], "the factory ran despite an incapable pairing"
        assert build_ci_solver("test_only_algorithm", backend="stub") is not None
    finally:
        unregister_algorithm("test_only_algorithm")


class _SamplerOnly:
    name = "sampler_only"
    version = "0"

    def capabilities(self):
        return frozenset(("sample",))


def test_an_algorithm_declaring_an_unknown_primitive_is_refused():
    with pytest.raises(ValueError, match="unknown primitive"):
        register_algorithm("test_only_bad", lambda **kw: None, ("tomography",))


def test_build_ci_solver_produces_a_working_solver():
    n, k = 6, 3
    h, eri = random_spinor_integrals(n, seed=101)
    solver = build_ci_solver("sqd", backend="stub", n_elec=k, shots=8000, seed=7,
                             strategy=ReferenceExcitationStrategy(k, theta=1.2))
    assert isinstance(solver, AdaptiveCISolver)
    assert abs(solver.solve(_Integrals(h, eri))[0] - _exact_spectrum(n, k, h, eri)[0]) < EXACT


# --- sizing and provenance ---------------------------------------------------------------------

@pytest.mark.parametrize("ndet,n_spinor,n_states", [(1, 4, 1), (137, 8, 3), (4096, 10, 2)])
def test_subspace_gb_is_exact_on_both_sides(ndet, n_spinor, n_states):
    """⚠ Bounded on both sides, so a sizing function that grows a safety factor fails."""
    from kuiva.util import resources

    expected = (determinant_memory_gb(ndet, n_states) + resources.rdm_gb(n_spinor, 2))
    assert subspace_gb(ndet, n_spinor, n_states) == pytest.approx(expected, rel=0.0,
                                                                 abs=1e-18)


def test_the_report_carries_the_provenance_of_the_sampling():
    """The provenance rule: an energy whose record does not say which device and how many shots
    produced it is not interpretable."""
    n, k = 6, 3
    h, eri = random_spinor_integrals(n, seed=111)
    solver = _covering_solver(k)
    solver.solve(_Integrals(h, eri))
    report = solver.report()
    assert report["algorithm"] == "sqd"
    assert report["coverage"] == pytest.approx(1.0)
    assert report["provenance"]["backend"] == "stub"
    assert report["provenance"]["shots"] == 8000
    assert report["provenance"]["seed"] == 7


# --- Stage 4: the whole loop ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_casscf_system():
    """A small two-component system with no chemistry in it — the same shape
    ``test_orbopt_events.py`` uses, so a Stage-4 failure is attributable to the solver rather
    than to a hard optimization."""
    rng = np.random.default_rng(3)
    nao = 6
    n = 2 * nao
    factors = ThreeIndexAO(l_packed=rng.standard_normal((3 * nao, nao * (nao + 1) // 2)),
                           nao=nao, origin="cholesky")
    h_ao = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    h_ao = 0.5 * (h_ao + h_ao.conj().T)
    _, c0 = np.linalg.eigh(h_ao)
    spaces = OrbitalSpaces.from_counts(n_inactive=2, n_active=4, n_orb=n)
    return factors, h_ao, np.ascontiguousarray(c0), spaces, 2


def test_sqd_drives_the_event_gated_optimizer_to_the_exact_casscf(synthetic_casscf_system):
    """Stage 4, on a surface the inner engine is known to converge on.

    ``SQDSolver`` goes into the **unchanged** :func:`optimize_orbitals_events`, and nothing in
    ``mcscf/`` knows a quantum computer was involved. With full coverage the sampled subspace
    is the complete CAS space, so the trajectory must land on the exact-CI CASSCF answer.
    """
    factors, h_ao, c0, spaces, n_elec = synthetic_casscf_system
    # 60 rather than 40: a budget with slack over the ~42 macro-iterations this
    # random-integral system takes since the exact-diagonal preconditioner (the early
    # trajectory is knife-edge sensitive to step details); the claim under test is the
    # exact/sampled agreement, not the iteration count.
    kwargs = dict(max_iter=60, mode="second-order", conv_grad=1e-6, report=False)
    exact = optimize_orbitals(factors, h_ao, c0, spaces,
                              FullCISolver(spaces.n_active, n_elec), **kwargs)
    solver = _covering_solver(n_elec)
    sampled = optimize_orbitals_events(factors, h_ao, c0, spaces, solver, **kwargs)

    assert exact.converged and sampled.converged
    assert solver.coverage() == pytest.approx(1.0)
    assert abs(sampled.energy - exact.energy) < 1e-9
    # ⚠ Converged here means a small gradient AND an event that refused — the honest fixed
    # point for an adaptive space, and a statement no gradient norm alone can make.
    assert sampled.event_stable


def test_a_truncated_sqd_casscf_stays_above_the_exact_one(synthetic_casscf_system):
    """A capped subspace is variational at fixed orbitals, and the CASSCF built on it cannot
    reach below the exact-CI CASSCF."""
    factors, h_ao, c0, spaces, n_elec = synthetic_casscf_system
    kwargs = dict(max_iter=40, mode="second-order", conv_grad=1e-6, report=False)
    exact = optimize_orbitals(factors, h_ao, c0, spaces,
                              FullCISolver(spaces.n_active, n_elec), **kwargs)
    solver = _covering_solver(n_elec, max_determinants=3)
    truncated = optimize_orbitals_events(factors, h_ao, c0, spaces, solver, **kwargs)
    assert solver.n_determinants == 3
    assert truncated.energy > exact.energy - 1e-9


@pytest.mark.slow
def test_sqd_casscf_reproduces_the_exact_casscf_on_a_real_molecule():
    """Stage 4's exit criterion on a real system: Ti(2+) 3d^2, CAS(2,10), SOC off.

    ⚠ Two separate claims, and the second is the interesting one:

    * with full coverage the SQD-CASSCF *is* the exact CASSCF (measured: 0.0 Eh apart);
    * with a **capped** subspace of 8 of the 45 determinants, the CI truncation is largely
      absorbed by the orbitals. At the exact CASSCF orbitals that 8-determinant space sits a
      few times 1e-4 Eh **above** the full CI; letting the optimizer re-optimize *for its own
      space* shrinks the remaining gap by more than an order of magnitude while staying above
      it, as a variational method must.

    ⚠ The test asserts the *ratio*, not either number. How far the relaxed gap falls depends
    on which eight determinants this seed happened to sample — measured between 0 and 7e-6 Eh
    for two nearby settings — so pinning an absolute value here would be pinning a draw. The
    reduction is the phenomenon; its size is a sample.

    ⚠ And do not read any of this as evidence that an 8-determinant CI is adequate. It is a
    single-state calculation, where a two-step CASSCF has the most freedom to hide a CI
    truncation in the orbitals; the state-averaged case (also measured during development, at
    3 roots) leaves a real 1e-2 Eh gap at the same cap.
    """
    # ⚠ The only test in this file that needs a front end, and therefore the only one that
    # cannot run under ``external/venv_qc`` — PySCF is kept out of that venv deliberately,
    # so that its NumPy 2.x never reaches the pinned-version baseline. Skipped there, run in the
    # default (slow) suite under ``external/venv``.
    pytest.importorskip("pyscf", reason="the front end is absent from external/venv_qc ")
    from kuiva.interface.api import Molecule, active_space_for, spinor_reference

    molecule = Molecule(atoms=[("Ti", (0.0, 0.0, 0.0))], basis="x2c-SVPall-2c",
                        charge=2, spin=2)
    reference = spinor_reference(molecule, memory_gb=4.0, screening="none", with_soc=False)
    active = active_space_for(reference, character=("Ti", "d"), n_active=10, n_active_elec=2)
    spaces, n_elec = active.spaces, active.n_elec
    h_ao, c0 = reference.h_one_electron(), reference.spinors_in_ao()
    kwargs = dict(e_nuc=reference.data.e_nuc, max_iter=40, mode="second-order",
                  conv_grad=1e-5, report=False)

    exact = optimize_orbitals(reference.factors, h_ao, c0, spaces,
                              FullCISolver(spaces.n_active, n_elec), **kwargs)
    assert exact.converged

    solver = _covering_solver(n_elec, shots=20000)
    full = optimize_orbitals_events(reference.factors, h_ao, c0, spaces, solver, **kwargs)
    assert full.converged and solver.coverage() == pytest.approx(1.0)
    assert abs(full.energy - exact.energy) < 1e-8

    # The truncation is real at fixed orbitals ...
    capped = _covering_solver(n_elec, shots=3000, max_determinants=8)
    frozen = capped.solve(CASIntegrals.build(reference.factors, h_ao, exact.coeff, spaces,
                                             e_nuc=reference.data.e_nuc))[0]
    frozen_gap = frozen - exact.energy
    assert frozen_gap > 1e-5

    # ... and the orbital optimizer largely absorbs it when it may re-optimize for that space.
    capped = _covering_solver(n_elec, shots=3000, max_determinants=8)
    truncated = optimize_orbitals_events(reference.factors, h_ao, c0, spaces, capped,
                                         **kwargs)
    relaxed_gap = truncated.energy - exact.energy
    assert capped.n_determinants == 8
    assert relaxed_gap > -1e-9, "a subspace CASSCF cannot go below the exact-CI one"
    assert relaxed_gap < frozen_gap / 5.0, (frozen_gap, relaxed_gap)
