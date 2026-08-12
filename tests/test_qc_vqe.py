"""Stage 5: the VQE path .

What this file can fail on that ``tests/test_qc_sqd.py`` cannot
----------------------------------------------------------------
⚠ **SQD never touches the qubit mapping.** Its subspace Hamiltonian is built by
``ci/strings.hamiltonian_matrix``, so every SQD test in this suite passes with
:mod:`kuiva.qc.mapping` arbitrarily wrong. VQE is the other half: its energy is
``sum_t c_t <P_t>`` over the mapped operator, measured through the backend's ``estimate``
primitive and the circuit compiler. A VQE that reaches ``FullCISolver``'s energy is therefore
an end-to-end check of mapping, circuits and estimator **together**, and it is the reason
keeps this path in scope at all.

The cheapest such check needs no optimizer, and is the first test below: for **one** electron
the two-body operator contributes nothing, the exact ground state is a single Slater
determinant, and :func:`kuiva.qc.fermionic.orbital_rotation_circuit` prepares it exactly. So
the exact answer is reachable by construction and any disagreement is a defect rather than an
optimization failure — which is what separates "the mapping is wrong" from "COBYLA stopped".

⚠ **The RDM tomography is the point of the expensive tests, not an implementation detail.**
The ``estimate`` route measures ``O(n^4)`` operators to obtain densities SQD gets free from its
classical diagonalization. Making that cost countable is what the scope decision asks for.
"""
import sys

import numpy as np
import pytest

sys.path.insert(0, "tests")

from kuiva.ci.strings import CASSpace, hamiltonian_matrix, popcount, rdm12
from kuiva.integrals.transform import ThreeIndexAO
from kuiva.mcscf.casci import FullCISolver
from kuiva.mcscf.orbopt import OrbitalSpaces, optimize_orbitals
from kuiva.qc.algorithms import algorithm_spec, build_ci_solver
from kuiva.qc.ansatz import HardwareEfficientStrategy, UCCStrategy
from kuiva.qc.backend import CapabilityError
from kuiva.qc.circuits import CircuitSpec
from kuiva.qc.fermionic import orbital_rotation_circuit
from kuiva.qc.mapping import qubit_hamiltonian, rdm_measurement
from kuiva.qc.vqe import SECTOR_WEIGHT_FLOOR, VQESolver
from test_ci_strings import random_spinor_integrals
from test_qc_sqd import _Integrals

#: Machine precision: the exact-estimator path evaluates the *same* operator the reference
#: diagonalizes, so anything above rounding is a defect.
EXACT = 1e-10


def _exact_ground(n, k, h, eri, e_core=0.0):
    return float(np.linalg.eigvalsh(
        hamiltonian_matrix(CASSpace(n, k).determinants(), h, eri).toarray())[0]) + e_core


# --- the optimizer-free end-to-end check -------------------------------------------------------

def test_the_exact_one_electron_state_is_prepared_and_measured_exactly():
    """⚠ The strongest cheap statement in this file: no optimizer, no tolerance to argue.

    With one electron the two-body operator has nothing to act on, so the exact ground state is
    the lowest eigenvector of ``h`` — a single Slater determinant. Compiling that orbital
    rotation and measuring the mapped Hamiltonian must return the eigenvalue itself, which
    exercises the mapping, the Givens network and the estimator in one line and can fail on
    any of them.
    """
    n = 5
    h, eri = random_spinor_integrals(n, seed=4)
    ints = _Integrals(h, eri, e_core=0.75)
    energies, vectors = np.linalg.eigh(h)

    solver = VQESolver(1, ansatz=UCCStrategy(1))
    operator = qubit_hamiltonian(ints)
    # ``vectors[:, 0]`` is the lowest orbital, so U(vectors) maps mode 0 onto it: prepare the
    # aufbau reference, then rotate.
    circuit = CircuitSpec.prepare(0b1, n).then(orbital_rotation_circuit(vectors).circuit)
    got = solver.energy_of(circuit, operator, ints.e_core)
    assert got == pytest.approx(energies[0] + ints.e_core, abs=EXACT)


# --- densities ----------------------------------------------------------------------------------

def test_the_two_rdm_routes_agree_and_reproduce_the_classical_builder():
    """``rdm_measurement`` (what hardware would do) against ``ci/strings.rdm12`` (what Kuiva
    does classically) on the *same* state. Independent code paths, exact comparison."""
    n, n_elec = 4, 2
    h, eri = random_spinor_integrals(n, seed=6)
    ints = _Integrals(h, eri)
    solver = VQESolver(n_elec, ansatz=UCCStrategy(n_elec), rdm_source="statevector")
    params = solver.ansatz.initial_parameters(n, h=h, eri=eri)

    gamma_sv, gamma2_sv = solver.rdms(ints, params)
    measured = VQESolver(n_elec, ansatz=solver.ansatz, rdm_source="estimate")
    measured.params = params
    gamma_est, gamma2_est = measured.rdms(ints, params)

    assert np.abs(gamma_sv - gamma_est).max() < EXACT
    assert np.abs(gamma2_sv - gamma2_est).max() < EXACT
    # And both satisfy the trace condition the classical RDMs are held to.
    assert np.trace(gamma_sv).real == pytest.approx(n_elec, abs=1e-10)
    assert np.abs(np.einsum("pqrr->pq", gamma2_sv)
                  - (n_elec - 1) * gamma_sv).max() < 1e-10


def test_the_rdm_measurement_plan_counts_the_tomography_cost():
    """⚠ The number this exists to make visible: SQD measures ``sample`` and nothing else."""
    plan = rdm_measurement(4, rank=2)
    assert plan.n_strings > 0
    assert plan.one_body.shape == (16, plan.n_strings)
    assert plan.two_body.shape == (256, plan.n_strings)
    with pytest.raises(ValueError, match="rank=1"):
        rdm_measurement(4, rank=1).gamma2(np.zeros(1))


def test_a_non_conserving_ansatz_warns_that_its_rdms_are_a_projection(kuiva_caplog):
    """⚠ "Proceeds but the user should know" is behaviour, so it is asserted as such.

    A hardware-efficient circuit puts most of its norm outside the particle-number sector, and
    the RDMs of the renormalized projection are not the densities of the state whose energy was
    reported.
    """
    n, n_elec = 4, 2
    h, eri = random_spinor_integrals(n, seed=8)
    ints = _Integrals(h, eri)
    solver = VQESolver(n_elec, ansatz=_LeakyAnsatz(n_elec))
    solver.rdms(ints, solver.ansatz.initial_parameters(n))
    assert any("sector" in record.getMessage() for record in kuiva_caplog.records)


class _LeakyAnsatz(HardwareEfficientStrategy):
    """The Stage-B circuit dressed as a variational ansatz, purely to leak particle number."""

    def n_parameters(self, n_qubit):
        return 0

    def initial_parameters(self, n_qubit, *, h=None, eri=None):
        return np.zeros(0)

    def build(self, n_qubit, params=None, *, h=None, eri=None):
        from kuiva.qc.fermionic import CompiledCircuit
        return CompiledCircuit(self.plan(n_qubit).circuits[0])


# --- capability negotiation -----------------------------------------------------------------------

def test_a_statevector_rdm_route_refuses_a_backend_that_cannot_provide_one():
    """The refusal culture: named at construction, never discovered mid-calculation."""

    class _SamplerOnly:
        name = "sampler_only"
        version = "0"

        def capabilities(self):
            return frozenset(("sample", "estimate"))

    with pytest.raises(CapabilityError) as excinfo:
        VQESolver(2, backend=_SamplerOnly(), rdm_source="statevector")
    for expected in ("vqe", "statevector", "sampler_only"):
        assert expected in str(excinfo.value)
    # ...and the same backend is fine once the RDMs are measured instead.
    VQESolver(2, backend=_SamplerOnly(), rdm_source="estimate")


def test_the_registry_declares_what_vqe_needs():
    spec = algorithm_spec("vqe")
    assert spec.requires == ("estimate",)
    solver = build_ci_solver("vqe", backend="stub", n_elec=2)
    assert isinstance(solver, VQESolver)


def test_a_finite_difference_gradient_is_refused():
    """⚠ Not an omission: on a shot-noisy energy its error is the noise over the step, which
    is unbounded as the step shrinks. Offering it would invite it onto hardware."""
    with pytest.raises(ValueError, match="parameter-shift"):
        VQESolver(2, gradient="finite-difference")


def test_lbfgs_refuses_to_run_without_the_shift_rule():
    n, n_elec = 4, 2
    h, eri = random_spinor_integrals(n, seed=9)
    with pytest.raises(ValueError, match="finite-difference"):
        VQESolver(n_elec, optimizer="l-bfgs-b").optimize(_Integrals(h, eri))


# --- variational behaviour --------------------------------------------------------------------

def test_the_exact_estimator_energy_is_a_variational_upper_bound():
    """True at **any** parameters, which is what makes it worth asserting: it is a statement
    about the estimator and the mapping, not about the optimizer."""
    n, n_elec = 4, 2
    h, eri = random_spinor_integrals(n, seed=12)
    ints = _Integrals(h, eri, e_core=-0.4)
    exact = _exact_ground(n, n_elec, h, eri, ints.e_core)
    solver = VQESolver(n_elec, ansatz=UCCStrategy(n_elec))
    operator = qubit_hamiltonian(ints)
    rng = np.random.default_rng(0)
    for _ in range(5):
        params = rng.normal(0.0, 0.6, size=solver.ansatz.n_parameters(n))
        circuit = solver.ansatz.build(n, params, h=h, eri=eri).circuit
        assert solver.energy_of(circuit, operator, ints.e_core) >= exact - EXACT


def test_a_particle_conserving_ansatz_keeps_the_whole_norm_in_the_sector():
    n, n_elec = 6, 3
    h, eri = random_spinor_integrals(n, seed=13)
    solver = VQESolver(n_elec, ansatz=UCCStrategy(n_elec))
    psi = solver.state(_Integrals(h, eri),
                       solver.ansatz.initial_parameters(n, h=h, eri=eri))
    inside = psi[popcount(np.arange(1 << n, dtype=np.uint64)) == n_elec]
    weight = float((np.abs(inside) ** 2).sum())
    assert weight > SECTOR_WEIGHT_FLOOR
    assert weight == pytest.approx(1.0, abs=1e-12)


# --- the expensive claims ------------------------------------------------------------------------

@pytest.mark.slow
def test_a_generalized_ucc_vqe_reaches_the_full_ci():
    """⚠ The end-to-end validation of the mapping layer, and the reason it needs the
    *generalized* ansatz: plain UCCSD from a fixed reference cannot reach a state that has
    almost no weight on that reference, which is the normal situation for the random integrals
    used throughout this suite. That is a statement about the ansatz, not about the mapping.
    """
    n, n_elec = 4, 2
    h, eri = random_spinor_integrals(n, seed=5)
    ints = _Integrals(h, eri, e_core=0.3)
    exact = _exact_ground(n, n_elec, h, eri, ints.e_core)
    solver = VQESolver(n_elec, ansatz=UCCStrategy(n_elec, generalized=True),
                       optimizer="cobyla", maxiter=2000)
    energy, _gamma, _gamma2 = solver.solve(ints)
    assert energy - exact < 1e-6
    assert energy >= exact - EXACT


@pytest.mark.slow
def test_a_vqe_drives_a_full_casscf_through_the_unchanged_optimizer():
    """The ``ci_solver`` contract, satisfied by a solver whose CI step is a measured quantum circuit.

    ⚠ The comparison is with the exact-CI CASSCF and the tolerance is loose on purpose: the
    VQE energy is converged only to what COBYLA reached, and that is a **noise floor** on
    ``E(kappa)`` exactly as the cheap CI's solver tolerance is. A tight tolerance here
    would be a claim about the classical optimizer, not about the pipeline.
    """
    rng = np.random.default_rng(3)
    nao, n_elec = 5, 2
    n = 2 * nao
    factors = ThreeIndexAO(l_packed=rng.standard_normal((3 * nao, nao * (nao + 1) // 2)),
                           nao=nao, origin="cholesky")
    h_ao = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    h_ao = 0.5 * (h_ao + h_ao.conj().T)
    _, c0 = np.linalg.eigh(h_ao)
    c0 = np.ascontiguousarray(c0)
    spaces = OrbitalSpaces.from_counts(n_inactive=2, n_active=4, n_orb=n)
    kwargs = dict(max_iter=25, mode="second-order", conv_grad=1e-5, report=False)

    exact = optimize_orbitals(factors, h_ao, c0, spaces,
                              FullCISolver(spaces.n_active, n_elec), **kwargs)
    solver = VQESolver(n_elec, ansatz=UCCStrategy(n_elec, generalized=True),
                       optimizer="cobyla", maxiter=1500)
    variational = optimize_orbitals(factors, h_ao, c0, spaces, solver, **kwargs)

    assert variational.energy >= exact.energy - 1e-8      # variational, up to the noise floor
    assert variational.energy - exact.energy < 1e-4
    assert solver.n_solves > 1
    assert solver.report()["provenance"]["backend"] == "stub"
