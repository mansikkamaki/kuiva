"""The circuit representation that crosses the hardware boundary — Kuiva's, not a vendor's.

**Orchestration and data, not a registered kernel.**

Why a kuiva-owned circuit type at all
-------------------------------------
The useful device is not known: VTT Q50 is the current target, access is call-gated, and the
device that matters in five years may be a different vendor, gate set or access model
entirely. So the rule is stronger than "wrap Qiskit cleanly": **no
algorithmic code in** :mod:`kuiva.qc` **may depend on a vendor stack**, so retargeting is an
adapter and never a rewrite. The object that crosses the boundary is therefore a
:class:`CircuitSpec` — gate names from a small fixed vocabulary, qubit indices, real
parameters — and never a ``QuantumCircuit``, a ``cirq.Circuit`` or anything like them, in
either direction.

Three consequences that are the point rather than side effects:

* the ansatz research — the genuinely novel part, generalizing
  spin-separable ansatz families to complex spinor excitations — lives in Kuiva and is
  portable to every backend ever written, for free;
* gate-set lowering, connectivity and calibration-aware routing are entirely the adapter's
  problem, quarantined exactly as the front-end quarantines contraction type in the front-end;
* only plain data crosses, so an **out-of-process** adapter (a vendor SDK with conflicting
  dependencies, a newer Python, a remote service) is never foreclosed. Nothing here may
  acquire a callback into Kuiva, shared mutable state or an open handle.

The vocabulary is deliberately minimal
--------------------------------------
Five single-qubit gates and one two-qubit entangler. A gate outside a device's native set is
the adapter's problem to decompose, not the algorithm's problem to avoid — so the vocabulary
grows only when an *ansatz* needs an operation that cannot be written with what is here, never
because some device happens to have one.

Conventions, both of them, stated because a silent mismatch here is a wrong statevector that
looks fine:

* **Statevector index is little-endian in qubit number**: amplitude ``j`` is the computational
  basis state whose bit ``k`` is qubit ``k``'s value. That is the same integer as a
  determinant mask and as :mod:`kuiva.qc.mapping`'s Pauli bookkeeping, which is what
  lets an occupation mask be prepared with no translation. It is also Qiskit's convention.
* **A gate's own matrix is written with** ``qubits[0]`` **as the most significant index** —
  the textbook ordering, in which ``cx(control, target)`` is the familiar
  ``[[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]``. This is a statement about the ``4 x 4``
  array only and has no bearing on the statevector layout above.

Single-qubit matrices follow the standard (and Qiskit's) definitions, ``r_a(theta) =
exp(-i theta a / 2)``, so a statevector produced here and one produced by an adapter agree
including global phase — which is what makes ``tests/test_qc_backend.py``'s stub-versus-Aer
comparison an equality rather than a similarity.

References
----------------------------
* Gate definitions and the little-endian statevector convention: M. A. Nielsen, I. L. Chuang,
  "Quantum Computation and Quantum Information", Cambridge (2010), ch. 4; A. Javadi-Abhari et
  al., "Quantum computing with Qiskit", arXiv:2405.08810 (2024), for the convention the first
  adapter must match.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np

#: ``name -> (qubits, parameters)``. The whole vocabulary; see the module docstring for the
#: rule governing when it may grow.
GATE_ARITY: Dict[str, Tuple[int, int]] = {
    "x": (1, 0),        # bit flip; also how an occupation mask is prepared
    "h": (1, 0),        # Hadamard
    "rx": (1, 1),       # exp(-i theta X / 2)
    "ry": (1, 1),       # exp(-i theta Y / 2)
    "rz": (1, 1),       # exp(-i theta Z / 2)
    "cx": (2, 0),       # controlled-X; qubits = (control, target)
}

_SQRT_HALF = 1.0 / np.sqrt(2.0)


def gate_matrix(name: str, params: Sequence[float] = ()) -> np.ndarray:
    """The unitary of one gate, ``(2^k, 2^k)`` complex.

    The single definition of what every gate in :data:`GATE_ARITY` *means*. The stub simulator
    applies exactly this; an adapter translates to its framework's gate of the same name, and
    ``tests/test_qc_backend.py`` is what checks the two agree.
    """
    key = str(name).lower()
    if key not in GATE_ARITY:
        raise ValueError("unknown gate {!r}; the vocabulary is {}".format(
            name, ", ".join(sorted(GATE_ARITY))))
    n_qubit, n_param = GATE_ARITY[key]
    if len(params) != n_param:
        raise ValueError("gate {!r} takes {} parameter(s), got {}".format(
            key, n_param, len(params)))
    if key == "x":
        return np.array([[0, 1], [1, 0]], dtype=np.complex128)
    if key == "h":
        return _SQRT_HALF * np.array([[1, 1], [1, -1]], dtype=np.complex128)
    if key == "cx":
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                        dtype=np.complex128)
    half = 0.5 * float(params[0])
    c, s = np.cos(half), np.sin(half)
    if key == "rx":
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)
    if key == "ry":
        return np.array([[c, -s], [s, c]], dtype=np.complex128)
    return np.array([[np.exp(-1j * half), 0], [0, np.exp(1j * half)]], dtype=np.complex128)


@dataclass(frozen=True)
class Gate:
    """One gate application: a name, the qubits it acts on, its real parameters.

    Immutable and hashable, so a :class:`CircuitSpec` can be a dictionary key (a backend
    caching transpilation results wants that) and no consumer can mutate a circuit another
    consumer is holding.
    """

    name: str
    qubits: Tuple[int, ...]
    params: Tuple[float, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name).lower())
        object.__setattr__(self, "qubits", tuple(int(q) for q in self.qubits))
        object.__setattr__(self, "params", tuple(float(p) for p in self.params))
        if self.name not in GATE_ARITY:
            raise ValueError("unknown gate {!r}; the vocabulary is {}".format(
                self.name, ", ".join(sorted(GATE_ARITY))))
        n_qubit, n_param = GATE_ARITY[self.name]
        if len(self.qubits) != n_qubit:
            raise ValueError("gate {!r} acts on {} qubit(s), got {}".format(
                self.name, n_qubit, self.qubits))
        if len(self.params) != n_param:
            raise ValueError("gate {!r} takes {} parameter(s), got {}".format(
                self.name, n_param, self.params))
        if len(set(self.qubits)) != len(self.qubits):
            raise ValueError("gate {!r} was given a repeated qubit: {}".format(
                self.name, self.qubits))
        if any(q < 0 for q in self.qubits):
            raise ValueError("qubit indices must be non-negative, got {}".format(self.qubits))

    def matrix(self) -> np.ndarray:
        return gate_matrix(self.name, self.params)

    def as_record(self) -> Tuple[str, Tuple[int, ...], Tuple[float, ...]]:
        return (self.name, self.qubits, self.params)

    def __repr__(self) -> str:
        args = ", ".join(str(q) for q in self.qubits)
        if self.params:
            args += "; " + ", ".join("{:.6g}".format(p) for p in self.params)
        return "{}({})".format(self.name, args)


@dataclass(frozen=True)
class CircuitSpec:
    """A circuit as plain data: a qubit count and an ordered tuple of :class:`Gate`.

    Deliberately *not* a builder. It carries no classical register, no measurement
    instructions and no layout: measurement is what a backend primitive *does*
    (:meth:`~kuiva.qc.backend.QuantumBackend.sample`), not something the algorithm layer
    decides, and layout is the adapter's business.

    The initial state is ``|0...0>``; :meth:`prepare` gives the occupation-mask preparation
    every fermionic ansatz starts from.
    """

    n_qubit: int
    gates: Tuple[Gate, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_qubit", int(self.n_qubit))
        object.__setattr__(self, "gates", tuple(self.gates))
        if self.n_qubit < 0:
            raise ValueError("n_qubit must be non-negative, got {}".format(self.n_qubit))
        for gate in self.gates:
            if not isinstance(gate, Gate):
                raise TypeError("a CircuitSpec holds Gate objects; got {}"
                                .format(type(gate).__name__))
            if any(q >= self.n_qubit for q in gate.qubits):
                raise ValueError("gate {!r} acts outside the {}-qubit register"
                                 .format(gate, self.n_qubit))

    @property
    def n_gates(self) -> int:
        return len(self.gates)

    def __len__(self) -> int:
        return len(self.gates)

    @classmethod
    def prepare(cls, mask: int, n_qubit: int) -> "CircuitSpec":
        """Prepare the computational basis state ``|mask>`` from ``|0...0>``.

        ``mask`` is an occupation bitmask in exactly the convention of ``ci/strings.py`` and
        of :mod:`kuiva.qc.mapping` — bit ``k`` set means mode ``k`` occupied — so a
        determinant goes to a circuit with no translation. That correspondence is the whole
        reason the boundary rules call the fit structural rather than convenient.

        Emitted as ``x`` gates rather than as a distinct "prepare" instruction: ``X`` is
        native on every device anyone would target, and one fewer thing for an adapter to
        special-case is worth more than the abstraction.
        """
        mask = int(mask)
        n_qubit = int(n_qubit)
        if mask < 0 or (n_qubit < 64 and mask >> n_qubit):
            raise ValueError("mask {:#x} sets a bit outside the {}-qubit register"
                             .format(mask, n_qubit))
        gates = tuple(Gate("x", (k,)) for k in range(n_qubit) if (mask >> k) & 1)
        return cls(n_qubit=n_qubit, gates=gates)

    def with_gate(self, name: str, qubits: Sequence[int],
                  params: Sequence[float] = ()) -> "CircuitSpec":
        """A new circuit with one gate appended. Immutable: the original is untouched."""
        return CircuitSpec(self.n_qubit,
                           self.gates + (Gate(name, tuple(qubits), tuple(params)),))

    def then(self, other: "CircuitSpec") -> "CircuitSpec":
        """Concatenation. Refuses circuits of different widths rather than padding one."""
        if other.n_qubit != self.n_qubit:
            raise ValueError("cannot concatenate a {}-qubit circuit with a {}-qubit one"
                             .format(self.n_qubit, other.n_qubit))
        return CircuitSpec(self.n_qubit, self.gates + other.gates)

    def to_records(self) -> Tuple[Tuple[str, Tuple[int, ...], Tuple[float, ...]], ...]:
        """The gate list as plain builtin tuples.

        Exists so a future out-of-process adapter stays possible without anyone having to
        prove it later: this and :meth:`from_records` are a serialization round trip using
        nothing but ``str``, ``int`` and ``float``.
        """
        return tuple(gate.as_record() for gate in self.gates)

    @classmethod
    def from_records(cls, n_qubit: int, records: Iterable) -> "CircuitSpec":
        return cls(n_qubit, tuple(Gate(name, tuple(qubits), tuple(params))
                                  for name, qubits, params in records))

    def gate_counts(self) -> Dict[str, int]:
        """How many of each gate — the cheapest honest cost figure for a circuit."""
        counts: Dict[str, int] = {}
        for gate in self.gates:
            counts[gate.name] = counts.get(gate.name, 0) + 1
        return counts

    def __repr__(self) -> str:
        counts = self.gate_counts()
        summary = ", ".join("{}={}".format(k, counts[k]) for k in sorted(counts))
        return "CircuitSpec(n_qubit={}, n_gates={}{})".format(
            self.n_qubit, self.n_gates, ", " + summary if summary else "")


def apply_circuit(circuit: CircuitSpec, psi: np.ndarray) -> np.ndarray:
    """Evolve a statevector by a circuit — the reference semantics of the vocabulary.

    Lives here, beside :func:`gate_matrix`, rather than in the stub backend: it *is* the
    definition of what a :class:`CircuitSpec` means, and a backend adapter is correct exactly
    insofar as it reproduces this. Straight tensor contraction, no optimization; the stub is a
    validation instrument and its cost is bounded by the ``2^n`` statevector long before it is
    bounded by this loop.
    """
    n = circuit.n_qubit
    if psi.size != (1 << n):
        raise ValueError("statevector of length {} does not match a {}-qubit circuit"
                         .format(psi.size, n))
    state = np.ascontiguousarray(psi, dtype=np.complex128).reshape((2,) * n)
    for gate in circuit.gates:
        k = len(gate.qubits)
        # Axis of qubit q under the little-endian index convention: qubit 0 is the least
        # significant bit, so it is the LAST tensor axis.
        axes = [n - 1 - q for q in gate.qubits]
        state = np.moveaxis(state, axes, range(k))
        shape = state.shape
        state = gate.matrix() @ state.reshape(1 << k, -1)
        state = np.moveaxis(state.reshape(shape), range(k), axes)
    return np.ascontiguousarray(state).reshape(-1)


__all__ = ["GATE_ARITY", "CircuitSpec", "Gate", "apply_circuit", "gate_matrix"]
