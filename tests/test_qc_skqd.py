"""Stage 5: sample-based Krylov quantum diagonalization .

What is actually new here, and what is deliberately shared
------------------------------------------------------------
SKQD is **not** a second algorithm stack. It is :class:`~kuiva.qc.sqd.SQDSolver` with a
different circuit source: instead of sampling an ansatz state it samples (Trotterized)
``exp(-iHt)|ref>`` for a ladder of times, whose span is approximately a Krylov space of ``H``.
Everything downstream — recovery, the subspace diagonalization, the RDMs, the adaptive-solver protocol (kuiva/mcscf/adaptive.py)
— is the same code, which is exactly the design argued for and the reason
this file is short.

So what is tested here is the *strategy*, plus the two structural claims the plan makes about
it:

* ⚠ **It is the first strategy that reads the integrals.** The Stage-A/B circuits are built
  from a mode count and a reference occupation, so under event gating a proposal adds
  statistical coverage and nothing else. A time-evolution circuit changes when the orbitals
  change, which is what makes a proposal a re-selection. Tested by *comparing circuits*, not by
  reading the docstring.
* **It sheds the ansatz-design problem** — there is no amplitude to guess, so the sampled
  distribution's concentration on low-lying configurations is a property of ``H``.

⚠ **Cost, not accuracy, is the binding constraint**, and the tests are sized accordingly: the
Hamiltonian has ``O(n^4)`` Pauli terms and each is a Pauli exponential of weight up to ``n``,
so a Trotter step is thousands of gates at six spinors and tens of thousands at ten. Nothing
in the default suite evolves anything larger than six.
"""
import sys

import numpy as np
import pytest

sys.path.insert(0, "tests")

from kuiva.ci.strings import CASSpace, hamiltonian_matrix, popcount
from kuiva.qc.algorithms import algorithm_spec, available_algorithms, build_ci_solver
from kuiva.qc.ansatz import TimeEvolutionStrategy
from kuiva.qc.sqd import SQDSolver
from test_ci_strings import random_spinor_integrals
from test_qc_sqd import _Integrals

EXACT = 1e-11


def _exact_ground(n, k, h, eri):
    return float(np.linalg.eigvalsh(
        hamiltonian_matrix(CASSpace(n, k).determinants(), h, eri).toarray())[0])


# --- the strategy -------------------------------------------------------------------------------

def test_the_time_evolution_strategy_refuses_to_plan_without_integrals():
    """⚠ The first strategy for which this is true — and the whole point of it."""
    with pytest.raises(ValueError, match="needs the integrals"):
        TimeEvolutionStrategy(2).plan(4)


def test_the_circuit_changes_when_the_integrals_do():
    """The claim that makes an event-gated proposal a re-selection rather than a redraw.

    Compared as *circuits*: the Stage-A and Stage-B strategies produce the identical circuit
    for two different Hamiltonians, and this one does not.
    """
    from kuiva.qc.ansatz import ReferenceExcitationStrategy

    n, n_elec = 4, 2
    h1, eri1 = random_spinor_integrals(n, seed=1)
    h2, eri2 = random_spinor_integrals(n, seed=2)
    strategy = TimeEvolutionStrategy(n_elec, times=(0.4,))
    first = strategy.plan(n, h=h1, eri=eri1).circuits[0]
    second = strategy.plan(n, h=h2, eri=eri2).circuits[0]
    assert first != second
    # The control: a Stage-A circuit is blind to both.
    stage_a = ReferenceExcitationStrategy(n_elec)
    assert stage_a.plan(n, h=h1, eri=eri1).circuits[0] == \
        stage_a.plan(n, h=h2, eri=eri2).circuits[0]


def test_the_ladder_produces_one_circuit_per_time_and_shares_the_shots():
    n, n_elec = 4, 2
    h, eri = random_spinor_integrals(n, seed=3)
    plan = TimeEvolutionStrategy(n_elec, times=(0.3, 0.6, 0.9)).plan(n, h=h, eri=eri)
    assert len(plan.circuits) == 3
    allocation = plan.allocate(900)
    assert sum(allocation) == 900 and min(allocation) > 0
    assert "SKQD" in plan.label


def test_screening_is_reported_as_a_truncation_of_the_hamiltonian(kuiva_caplog):
    """⚠ Dropping small Pauli coefficients is a truncation, not a tidy-up, so it warns."""
    n, n_elec = 4, 2
    h, eri = random_spinor_integrals(n, seed=4)
    dense = TimeEvolutionStrategy(n_elec, times=(0.3,)).plan(n, h=h, eri=eri)
    sparse = TimeEvolutionStrategy(n_elec, times=(0.3,), screen=0.05).plan(n, h=h, eri=eri)
    assert sparse.circuits[0].n_gates < dense.circuits[0].n_gates
    assert any("truncation of the Hamiltonian" in record.getMessage()
               for record in kuiva_caplog.records)


def test_more_trotter_steps_make_a_deeper_circuit_for_the_same_time():
    n, n_elec = 4, 2
    h, eri = random_spinor_integrals(n, seed=5)
    coarse = TimeEvolutionStrategy(n_elec, times=(2.0,), steps=1).plan(n, h=h, eri=eri)
    fine = TimeEvolutionStrategy(n_elec, times=(2.0,), steps=3).plan(n, h=h, eri=eri)
    # The reference preparation is emitted once, the slices three times over.
    prepared = n_elec
    assert (fine.circuits[0].n_gates - prepared
            == 3 * (coarse.circuits[0].n_gates - prepared))


def test_a_pauli_trotter_evolution_leaks_particle_number_and_the_leak_falls_with_steps():
    """⚠ The counter-intuitive property, pinned as a measurement rather than left as prose.

    ``H`` commutes with ``N`` and so does ``exp(-iHt)``; the individual **Pauli strings** ``H``
    decomposes into do not, so a product of their exponentials leaks off the sector. A reader
    who reasoned "the Hamiltonian conserves N, therefore the circuit does" would predict a
    100% recovery yield for SKQD, and would be wrong. The leak is an ordinary Trotter error and
    vanishes with the slice count, which is what the ladder below establishes; removing it
    outright needs the number-conserving decomposition ``kuiva/qc/fermionic.py`` defers.
    """
    from kuiva.qc.backends.stub import StubBackend

    n, n_elec = 4, 2
    h, eri = random_spinor_integrals(n, seed=6)
    sector = popcount(np.arange(1 << n, dtype=np.uint64)) == n_elec
    leaks = []
    for slices in (1, 2, 4, 8):
        strategy = TimeEvolutionStrategy(n_elec, times=(0.5,), steps=2 * slices)
        psi = StubBackend().statevector(strategy.plan(n, h=h, eri=eri).circuits[0])
        leaks.append(1.0 - float((np.abs(psi[sector]) ** 2).sum()))
    assert leaks[0] > 1e-2, "one slice must show the effect, or the test is vacuous"
    assert all(fine < coarse for coarse, fine in zip(leaks[:-1], leaks[1:]))
    assert leaks[-1] < 1e-4


# --- the algorithm registry entry ------------------------------------------------------------------

def test_skqd_is_registered_and_is_the_sqd_driver():
    assert "skqd" in available_algorithms()
    assert algorithm_spec("skqd").requires == ("sample",)
    solver = build_ci_solver("skqd", backend="stub", n_elec=2, times=(0.4, 0.8))
    assert isinstance(solver, SQDSolver)
    assert isinstance(solver.strategy, TimeEvolutionStrategy)
    assert solver.strategy.times == (0.4, 0.8)


def test_skqd_refuses_a_strategy_argument():
    """⚠ The name *is* the circuit source. A ``strategy=`` would make it describe a run it did
    not perform, which is the failure ``resolve_strategy`` refuses one level down."""
    from kuiva.qc.ansatz import ReferenceExcitationStrategy

    with pytest.raises(ValueError, match="would make the name mean something else"):
        build_ci_solver("skqd", backend="stub", n_elec=2,
                        strategy=ReferenceExcitationStrategy(2))


# --- end to end ----------------------------------------------------------------------------------

def test_the_skqd_subspace_is_variational_and_reproduces_the_ci_when_it_is_complete():
    n, n_elec = 4, 2
    h, eri = random_spinor_integrals(n, seed=7)
    ints = _Integrals(h, eri, e_core=0.25)
    exact = _exact_ground(n, n_elec, h, eri) + ints.e_core
    solver = build_ci_solver("skqd", backend="stub", n_elec=n_elec, times=(0.5, 1.0),
                             shots=4000, seed=11)
    energy, _gamma, _gamma2 = solver.solve(ints)
    assert energy >= exact - EXACT
    if solver.coverage() == pytest.approx(1.0):
        assert energy == pytest.approx(exact, abs=EXACT)


@pytest.mark.slow
def test_accumulating_skqd_proposals_lower_the_energy_monotonically():
    """⚠ Nested spaces, not a shot ladder: accumulation across proposals is what is monotone
    as a theorem (``kuiva/qc/sqd.py``), and a Krylov ladder is no exception to that."""
    n, n_elec = 6, 3
    h, eri = random_spinor_integrals(n, seed=8)
    ints = _Integrals(h, eri)
    exact = _exact_ground(n, n_elec, h, eri)
    solver = build_ci_solver("skqd", backend="stub", n_elec=n_elec, times=(0.4, 0.8),
                             steps=1, shots=2000, seed=3, max_determinants=8)
    energies = [solver.solve(ints)[0]]
    for _ in range(4):
        proposal = solver.propose(ints)
        if proposal is None:
            continue
        solver.adopt(proposal.key)
        energies.append(solver.solve(ints)[0])
    assert all(later <= earlier + 1e-12
               for earlier, later in zip(energies[:-1], energies[1:]))
    assert energies[-1] >= exact - EXACT
