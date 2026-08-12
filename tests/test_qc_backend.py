"""Stage 2: the hardware boundary — capabilities, the stub, and Aer .

Two tiers in one file, deliberately, because they can fail on completely different things and
the second is what tells you which:

* **The stub tier runs unconditionally.** ``kuiva.qc.backends.stub`` is Kuiva's own exact
  statevector simulator and needs nothing installed, so the whole boundary — capability
  negotiation, provenance, the shape and dtype of every result — is exercised in the default
  laptop suite. It is also, per ``amf/backend.py``'s precedent, the
  second implementation without which the protocol is indistinguishable from no protocol. ⚠
  It is never deleted for looking trivial, and it is never a fallback: nothing selects it
  because something else failed.
* **The Aer tier carries the ``qc`` marker** and is deselected by default (the suite's ``addopts``),
  because it needs ``external/venv_qc`` and a *different interpreter*. Run it with
  ``external/venv_qc/bin/python -m pytest -m qc``, with the repository root on ``PYTHONPATH``.

What the Aer tier can fail on that the stub tier cannot
---------------------------------------------------------
Everything about the *translation*: endianness of a measured bitstring, the direction of a
``cx``, the sign of a rotation, and — the one worth the most — Kuiva's ``i^popcount(x & z)``
Pauli phase convention, checked against Qiskit's own ``Pauli``/``Statevector`` rather than
against another copy of Kuiva's formula. A check whose two sides share an implementation
cannot see an error in it, and this is the only place in :mod:`kuiva.qc` where a second
implementation of the same quantity actually exists.
"""
import numpy as np
import pytest

from kuiva.qc.backend import (PRIMITIVES, BackendProvenance, CapabilityError, EstimateResult,
                              SampleResult, available_backends, get_backend,
                              register_backend, require_primitives, unregister_backend)
from kuiva.qc.backends.stub import StubBackend
from kuiva.qc.circuits import CircuitSpec
from kuiva.qc.mapping import jordan_wigner
from test_ci_strings import random_spinor_integrals

EXACT = 1e-11

#: Sampling tolerance at 200k shots. Three standard deviations of a proportion is at most
#: ``3 * 0.5 / sqrt(2e5) = 3.4e-3``; 1e-2 is comfortably above that and still tight enough to
#: catch a permuted or mislabelled outcome, which is what this kind of test is *for*.
SHOTS = 200_000
STATISTICAL = 1e-2


def _bell(n=2):
    return CircuitSpec(n).with_gate("h", (0,)).with_gate("cx", (0, 1))


def _entangled(n=4, seed=0):
    """A circuit with no structure a convention error could hide behind: every qubit rotated
    by a different angle about a different axis, then a full ``cx`` ladder."""
    rng = np.random.default_rng(seed)
    circuit = CircuitSpec(n)
    for axis in ("ry", "rz", "rx"):
        for q in range(n):
            circuit = circuit.with_gate(axis, (q,), (float(rng.normal()),))
        for q in range(n - 1):
            circuit = circuit.with_gate("cx", (q, q + 1))
    return circuit


# --- capability negotiation (no backend needed) --------------------------------------------

class _SamplerOnly:
    """A backend declaring one primitive. Exists to make the refusal testable without
    pretending some real device lacks a feature."""

    name = "sampler_only"
    version = "0"

    def capabilities(self):
        return frozenset(("sample",))


def test_an_incapable_pairing_is_refused_naming_both_sides():
    """⚠ Refused at construction, never emulated or degraded (the gate philosophy).

    The message is the feature: whoever hits it is choosing between names in two registries
    and needs to know which pairing exists, not that something is unsupported.
    """
    with pytest.raises(CapabilityError) as excinfo:
        require_primitives(_SamplerOnly(), ("sample", "statevector"), algorithm="vqe")
    message = str(excinfo.value)
    for expected in ("vqe", "statevector", "sampler_only", "sample"):
        assert expected in message, (expected, message)


def test_a_capable_pairing_passes_silently():
    require_primitives(_SamplerOnly(), ("sample",), algorithm="sqd")
    require_primitives(StubBackend(), PRIMITIVES, algorithm="everything")


def test_an_unknown_primitive_is_a_programming_error_not_a_capability_gap():
    with pytest.raises(ValueError, match="unknown primitive"):
        require_primitives(StubBackend(), ("tomography",), algorithm="wishful")


# --- the registry ---------------------------------------------------------------------------

def test_both_shipped_backends_are_registered_but_neither_is_imported_eagerly():
    """Registration says a backend is *known*, not that its framework is installed — which is
    exactly why ``import kuiva.qc.backend`` must pull in neither a simulator nor Qiskit."""
    import sys

    assert "stub" in available_backends()
    assert "qiskit_aer" in available_backends()
    assert "qiskit" not in sys.modules or "kuiva" not in repr(sys.modules["qiskit"])


def test_get_backend_refuses_an_unknown_name_and_lists_what_exists():
    with pytest.raises(ValueError, match="unknown quantum backend"):
        get_backend("dwave")


def test_a_registered_backend_is_what_get_backend_returns():
    register_backend("test_only_backend", lambda **kw: _SamplerOnly())
    try:
        assert isinstance(get_backend("test_only_backend"), _SamplerOnly)
    finally:
        unregister_backend("test_only_backend")
    assert "test_only_backend" not in available_backends()


# --- provenance ------------------------------------------------------------------------------

def test_provenance_travels_with_every_result_and_is_json_plain():
    import json

    backend = StubBackend()
    sampled = backend.sample(_bell(), 100, seed=7)
    assert sampled.provenance.backend == "stub"
    assert sampled.provenance.shots == 100 and sampled.provenance.seed == 7
    assert not sampled.provenance.exact
    assert json.loads(json.dumps(sampled.provenance.as_dict())) \
        == sampled.provenance.as_dict()

    exact = backend.estimate(_bell(), np.array([0], np.uint64), np.array([3], np.uint64))
    assert exact.provenance.exact and exact.provenance.shots is None


def test_an_unseeded_run_records_no_seed_rather_than_a_number_it_did_not_use():
    """⚠ A provenance claiming a seed the run did not use is worse than one admitting it is
    not replayable."""
    result = StubBackend().sample(_bell(), 32)
    assert result.provenance.seed is None


def test_a_seeded_simulator_result_is_replayable():
    """The provenance rule's promise, asserted rather than described."""
    a = StubBackend().sample(_entangled(3, seed=1), 5000, seed=42)
    b = StubBackend(seed=42).sample(_entangled(3, seed=1), 5000)
    assert np.array_equal(a.masks, b.masks) and np.array_equal(a.counts, b.counts)


def test_the_result_containers_refuse_malformed_data():
    prov = BackendProvenance()
    with pytest.raises(ValueError, match="unique ascending masks"):
        SampleResult(np.array([3, 1], np.uint64), np.array([1, 1], np.int64), prov)
    with pytest.raises(ValueError, match="unique ascending masks"):
        SampleResult(np.array([1, 3], np.uint64), np.array([1, 0], np.int64), prov)
    with pytest.raises(ValueError, match="matching 1-D arrays"):
        EstimateResult(np.zeros(3), np.zeros(2), prov)


def test_combine_weights_the_variances_by_the_squared_coefficients():
    prov = BackendProvenance(shots=1000)
    result = EstimateResult(np.array([0.5, -0.25]), np.array([1e-3, 4e-3]), prov)
    value, variance = result.combine(np.array([2.0, 3.0]))
    assert value == pytest.approx(2 * 0.5 + 3 * -0.25)
    assert variance == pytest.approx(4 * 1e-3 + 9 * 4e-3)


# --- the stub, against something that is not itself -----------------------------------------

def test_the_stub_statevector_is_the_circuit_semantics():
    from kuiva.qc.circuits import apply_circuit

    circuit = _entangled(4, seed=2)
    psi = np.zeros(16, dtype=np.complex128)
    psi[0] = 1.0
    assert np.abs(StubBackend().statevector(circuit) - apply_circuit(circuit, psi)).max() \
        < EXACT


def test_stub_sampling_reproduces_the_exact_distribution():
    backend = StubBackend()
    circuit = _entangled(4, seed=3)
    exact = np.abs(backend.statevector(circuit)) ** 2
    result = backend.sample(circuit, SHOTS, seed=11)
    observed = np.zeros_like(exact)
    observed[result.masks.astype(np.int64)] = result.probabilities()
    assert result.shots == SHOTS
    assert np.abs(observed - exact).max() < STATISTICAL


def test_stub_estimates_are_exact_without_shots_and_unbiased_with_them():
    backend = StubBackend()
    circuit = _entangled(3, seed=4)
    h, eri = random_spinor_integrals(3, seed=5)
    ps = jordan_wigner(h, eri)
    exact = backend.estimate(circuit, ps.x_masks, ps.z_masks)
    assert np.all(exact.variances == 0.0)
    assert np.abs(exact.values).max() <= 1.0 + EXACT

    sampled = backend.estimate(circuit, ps.x_masks, ps.z_masks, shots=SHOTS, seed=13)
    assert np.abs(sampled.values - exact.values).max() < STATISTICAL
    # ⚠ The variance is the variance OF THE ESTIMATOR: it carries the 1/shots.
    assert np.all(sampled.variances <= 1.0 / SHOTS + EXACT)


def test_the_stub_energy_of_a_prepared_determinant_is_its_diagonal_element():
    """The end-to-end statement the whole boundary exists for, on the one state where the
    answer is known independently: preparing ``|mask>`` and estimating the mapped Hamiltonian
    must give ``<I|H|I>``, which ``ci/strings.diagonal_energies`` computes by an entirely
    different route.
    """
    from kuiva.ci.strings import Determinants, diagonal_energies

    n, mask = 6, 0b010110
    h, eri = random_spinor_integrals(n, seed=19)
    ps = jordan_wigner(h, eri)
    backend = StubBackend()
    result = backend.estimate(CircuitSpec.prepare(mask, n), ps.x_masks, ps.z_masks)
    energy, _ = result.combine(ps.coeffs)
    dets = Determinants(np.array([mask], dtype=np.uint64), n, bin(mask).count("1"))
    assert abs(energy - float(diagonal_energies(dets, h, eri)[0])) < EXACT


def test_the_stub_refuses_a_non_positive_shot_count():
    with pytest.raises(ValueError, match="shots must be positive"):
        StubBackend().sample(_bell(), 0)


# --- Aer: the only place a second implementation of the same quantity exists -----------------

@pytest.fixture(scope="module")
def aer():
    """The Aer backend, or a skip that says how to install it."""
    from kuiva.qc import gate

    if not gate.available("qiskit_aer"):
        pytest.skip("qiskit_aer is absent; {}".format(gate.INSTALL_HINT))
    return get_backend("qiskit_aer", seed=1234)


@pytest.mark.qc
def test_aer_declares_the_primitives_it_has(aer):
    assert aer.capabilities() == frozenset(PRIMITIVES)
    assert "qiskit" in aer.version
    matrix_product = get_backend("qiskit_aer", method="matrix_product_state")
    assert "statevector" not in matrix_product.capabilities()
    with pytest.raises(CapabilityError, match="statevector"):
        require_primitives(matrix_product, ("statevector",), algorithm="exact-validation")


@pytest.mark.qc
@pytest.mark.parametrize("n,seed", [(2, 0), (3, 1), (4, 2), (5, 3)])
def test_aer_and_the_stub_agree_on_the_statevector(aer, n, seed):
    """Up to a global phase — the physically meaningful invariant, and immune to a transpiler
    that inserts one. The gate matrices are nevertheless defined to match Qiskit's exactly, so
    the phase found below should be 1; that is asserted separately rather than assumed."""
    circuit = _entangled(n, seed=seed)
    mine = StubBackend().statevector(circuit)
    theirs = aer.statevector(circuit)
    overlap = np.vdot(mine, theirs)
    assert abs(abs(overlap) - 1.0) < 1e-10
    assert np.abs(theirs - overlap / abs(overlap) * mine).max() < 1e-10
    assert abs(overlap - 1.0) < 1e-10, "an unexpected global phase entered the translation"


@pytest.mark.qc
@pytest.mark.parametrize("mask,n", [(0b0, 3), (0b101, 3), (0b1010, 4), (0b11011, 5)])
def test_aer_agrees_on_which_bit_is_which_qubit(aer, mask, n):
    """⚠ The endianness check, on both primitives at once.

    A measured bitstring must decode to the *same* integer a determinant carries. This is the
    single most likely convention error in an adapter and the one that produces a completely
    plausible wrong answer downstream: a permuted configuration list still diagonalizes.
    """
    circuit = CircuitSpec.prepare(mask, n)
    assert int(np.argmax(np.abs(aer.statevector(circuit)))) == mask
    sampled = aer.sample(circuit, 64, seed=3)
    assert sampled.masks.tolist() == [mask] and sampled.counts.tolist() == [64]


@pytest.mark.qc
def test_aer_and_the_stub_agree_on_a_sampled_distribution(aer):
    circuit = _entangled(4, seed=5)
    mine = StubBackend().sample(circuit, SHOTS, seed=21)
    theirs = aer.sample(circuit, SHOTS, seed=21)
    p_mine = np.zeros(16)
    p_mine[mine.masks.astype(np.int64)] = mine.probabilities()
    p_theirs = np.zeros(16)
    p_theirs[theirs.masks.astype(np.int64)] = theirs.probabilities()
    # Different RNGs, so this is two independent draws from the same distribution.
    assert np.abs(p_mine - p_theirs).max() < STATISTICAL
    assert theirs.shots == SHOTS


@pytest.mark.qc
def test_aer_confirms_kuivas_pauli_phase_convention(aer):
    """⚠ The check that is worth the most in this file.

    Kuiva's ``P(x, z) = i^popcount(x & z) X^x Z^z`` encoding, its ``Y = XZ`` phase and its
    qubit-0-first label order are all validated here against **Qiskit's** ``Pauli`` and
    ``Statevector`` — a genuinely independent implementation of the same operator algebra.
    Everywhere else in :mod:`kuiva.qc` the two sides of a Pauli check share Kuiva's formula,
    and such a check cannot fail on an error in it.
    """
    n = 4
    h, eri = random_spinor_integrals(n, seed=41)
    ps = jordan_wigner(h, eri)
    circuit = _entangled(n, seed=6)
    mine = StubBackend().estimate(circuit, ps.x_masks, ps.z_masks)
    theirs = aer.estimate(circuit, ps.x_masks, ps.z_masks)
    assert np.abs(mine.values - theirs.values).max() < 1e-10

    # And the energy those terms add up to is the CI expectation value of the same state.
    psi = StubBackend().statevector(circuit)
    dense = ps.to_dense()
    for result in (mine, theirs):
        energy, _ = result.combine(ps.coeffs)
        assert abs(energy - float(np.real(np.vdot(psi, dense @ psi)))) < 1e-10


@pytest.mark.qc
def test_aer_measures_pauli_strings_rather_than_resampling_exact_values(aer):
    """With ``shots``, the adapter rotates each Pauli into the computational basis and
    measures it. The estimate must be unbiased and its variance must fall as ``1/shots``."""
    n = 3
    h, eri = random_spinor_integrals(n, seed=43)
    ps = jordan_wigner(h, eri).drop_below(1e-2)
    circuit = _entangled(n, seed=7)
    exact = aer.estimate(circuit, ps.x_masks, ps.z_masks)
    sampled = aer.estimate(circuit, ps.x_masks, ps.z_masks, shots=20_000, seed=5)
    assert np.abs(sampled.values - exact.values).max() < 5e-2
    assert np.all(sampled.variances <= 1.0 / 20_000 + EXACT)
    assert sampled.provenance.shots == 20_000 and sampled.provenance.seed == 5


@pytest.mark.qc
def test_the_translated_circuit_is_inspectable(aer):
    """``to_qiskit`` is public because a disagreement between two backends has to be
    bisectable, and a translation nobody can look at is where a convention error hides."""
    circuit = _bell().with_gate("rz", (1,), (0.3,))
    translated = aer.to_qiskit(circuit)
    assert translated.num_qubits == 2
    assert [instruction.operation.name for instruction in translated.data] \
        == ["h", "cx", "rz"]


@pytest.mark.qc
def test_aer_measures_commuting_groups_together_and_stays_unbiased(aer):
    """Stage 5's grouping: fewer circuits, the same unbiased values.

    ⚠ Both halves matter. Fewer circuits is the whole point — it is what makes a VQE over
    thousands of Pauli terms affordable — but a grouping that got the bit bookkeeping wrong
    would *also* be fewer circuits, and would return plausible numbers. So the values are
    checked against the exact ones, and the recorded group structure against the count.
    """
    from kuiva.qc.mapping import qwc_groups

    n = 4
    h, eri = random_spinor_integrals(n, seed=51)
    ps = jordan_wigner(h, eri).drop_below(5e-2)
    circuit = _entangled(n, seed=9)
    groups = qwc_groups(ps)
    assert len(groups) < ps.n_terms, "a grouping that saves nothing proves nothing"

    exact = aer.estimate(circuit, ps.x_masks, ps.z_masks)
    grouped = aer.estimate(circuit, ps.x_masks, ps.z_masks, shots=40_000, seed=17)
    assert grouped.n_circuits == len(groups)
    assert exact.n_circuits == 0
    assert np.abs(grouped.values - exact.values).max() < 5e-2
    # ⚠ The recorded groups are what tells a caller that `combine`'s independence assumption
    # no longer holds; a result that silently dropped them would be the dangerous version.
    assert grouped.groups is not None
    assert sorted(int(t) for g in grouped.groups for t in g) == list(range(ps.n_terms))
