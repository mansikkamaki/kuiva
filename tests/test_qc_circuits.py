"""Stage 2: the circuit representation that crosses the hardware boundary .

A :class:`CircuitSpec` is plain data, so what there is to test is not arithmetic but the two
things that would be expensive to discover later: that the **conventions** are what the
docstring says (a statevector index that disagrees with a determinant mask by so much as an
endianness is a wrong answer that looks fine), and that the object really is serializable
plain data, so the out-of-process adapter the boundary rules keep open is not quietly foreclosed.

No framework is involved and none may be: this file runs in the default suite.
"""
import numpy as np
import pytest

from kuiva.qc.circuits import (GATE_ARITY, CircuitSpec, Gate, apply_circuit, gate_matrix)

EXACT = 1e-12


def _zero_state(n):
    psi = np.zeros(1 << n, dtype=np.complex128)
    psi[0] = 1.0
    return psi


def _run(circuit):
    return apply_circuit(circuit, _zero_state(circuit.n_qubit))


# --- the conventions ------------------------------------------------------------------------

@pytest.mark.parametrize("mask,n", [(0b0, 3), (0b1, 3), (0b101, 3), (0b11, 2), (0b1010, 4)])
def test_prepare_puts_the_amplitude_at_the_occupation_mask(mask, n):
    """⚠ The load-bearing convention of this whole package.

    ``CircuitSpec.prepare(mask)`` must put the amplitude at index ``mask`` — the *same*
    integer a determinant carries in ``ci/strings.py`` and the Pauli bookkeeping of
    ``kuiva.qc.mapping``. An endianness disagreement here would silently permute every
    sampled configuration, which is a wrong CI space that still diagonalizes to a plausible
    number.
    """
    psi = _run(CircuitSpec.prepare(mask, n))
    expected = np.zeros(1 << n, dtype=np.complex128)
    expected[mask] = 1.0
    assert np.abs(psi - expected).max() < EXACT


def test_cx_control_is_the_first_qubit():
    """``cx(control, target)``, asserted on the state rather than read off the matrix."""
    prepared = CircuitSpec.prepare(0b01, 2).with_gate("cx", (0, 1))
    assert np.argmax(np.abs(_run(prepared))) == 0b11
    # Control clear: nothing happens.
    prepared = CircuitSpec.prepare(0b10, 2).with_gate("cx", (0, 1))
    assert np.argmax(np.abs(_run(prepared))) == 0b10


def test_a_bell_pair_is_a_bell_pair():
    bell = CircuitSpec(2).with_gate("h", (0,)).with_gate("cx", (0, 1))
    psi = _run(bell)
    expected = np.zeros(4, dtype=np.complex128)
    expected[0b00] = expected[0b11] = 1.0 / np.sqrt(2.0)
    assert np.abs(psi - expected).max() < EXACT


@pytest.mark.parametrize("name", ["rx", "ry", "rz"])
def test_the_rotations_are_the_standard_exponentials(name):
    """``r_a(theta) = exp(-i theta a / 2)``, including global phase — the convention that
    makes an adapter's statevector comparable to the stub's rather than merely similar."""
    from scipy.linalg import expm

    pauli = {"rx": np.array([[0, 1], [1, 0]]),
             "ry": np.array([[0, -1j], [1j, 0]]),
             "rz": np.array([[1, 0], [0, -1]])}[name]
    theta = 0.7
    assert np.abs(gate_matrix(name, (theta,))
                  - expm(-0.5j * theta * pauli)).max() < EXACT


def test_every_gate_in_the_vocabulary_is_unitary():
    for name, (n_qubit, n_param) in GATE_ARITY.items():
        m = gate_matrix(name, (0.37,) * n_param)
        assert m.shape == (1 << n_qubit, 1 << n_qubit), name
        assert np.abs(m.conj().T @ m - np.eye(1 << n_qubit)).max() < EXACT, name


def test_a_circuit_preserves_the_norm():
    rng = np.random.default_rng(0)
    circuit = CircuitSpec(4)
    for _ in range(20):
        circuit = circuit.with_gate("ry", (int(rng.integers(4)),), (float(rng.normal()),))
        a, b = rng.choice(4, size=2, replace=False)
        circuit = circuit.with_gate("cx", (int(a), int(b)))
    assert abs(np.linalg.norm(_run(circuit)) - 1.0) < EXACT


# --- plain data -----------------------------------------------------------------------------

def test_the_gate_list_round_trips_through_builtin_tuples():
    """⚠ The out-of-process adapter is kept possible by construction, not by
    intention: the whole circuit must survive a trip through ``str``/``int``/``float``."""
    original = (CircuitSpec.prepare(0b1001, 4)
                .with_gate("h", (0,))
                .with_gate("cx", (0, 2))
                .with_gate("rz", (3,), (0.25,)))
    records = original.to_records()
    assert all(isinstance(name, str) and all(isinstance(q, int) for q in qubits)
               and all(isinstance(p, float) for p in params)
               for name, qubits, params in records)
    assert CircuitSpec.from_records(4, records) == original


def test_a_circuit_is_immutable_and_hashable():
    circuit = CircuitSpec.prepare(0b11, 2)
    extended = circuit.with_gate("h", (0,))
    assert circuit.n_gates == 2 and extended.n_gates == 3
    assert hash(circuit) != hash(extended)


# --- refusals -------------------------------------------------------------------------------

def test_an_unknown_gate_is_refused_with_the_vocabulary():
    with pytest.raises(ValueError, match="unknown gate"):
        Gate("iswap", (0, 1))


@pytest.mark.parametrize("args,match", [
    (("cx", (0,)), "acts on 2 qubit"),
    (("rz", (0,), ()), "takes 1 parameter"),
    (("h", (0,), (0.1,)), "takes 0 parameter"),
    (("cx", (1, 1)), "repeated qubit"),
    (("x", (-1,)), "non-negative"),
])
def test_malformed_gates_are_refused_at_construction(args, match):
    with pytest.raises(ValueError, match=match):
        Gate(*args)


def test_a_gate_outside_the_register_is_refused():
    with pytest.raises(ValueError, match="outside the 2-qubit register"):
        CircuitSpec(2, (Gate("x", (5,)),))


def test_prepare_refuses_a_mask_wider_than_the_register():
    with pytest.raises(ValueError, match="outside the 2-qubit register"):
        CircuitSpec.prepare(0b100, 2)


def test_concatenation_refuses_mismatched_widths():
    with pytest.raises(ValueError, match="cannot concatenate"):
        CircuitSpec(2).then(CircuitSpec(3))
