"""Qiskit Aer adapter — the first real backend, and the shape every later one copies.

**Orchestration, not a registered kernel.** Nothing here is algorithmic: it translates a
:class:`~kuiva.qc.circuits.CircuitSpec` into Qiskit's circuit type, runs it, and turns the
result back into the plain arrays of :mod:`kuiva.qc.backend`. Every decision that could be
called physics or method lives one layer up, which is the entire point of the boundary.

Why Qiskit first, and what that does *not* commit to
--------------------------------------------------------------------------
Qiskit is the first **adapter**, not the substrate. The mapping, the circuit representation,
the ansatz builders and the sampled-subspace driver are pure NumPy/SciPy; if the ecosystem
shifts, or a target device ships a non-Qiskit access stack, the cost is one new adapter and
everything validated against the stub and Aer transfers. Qiskit is first because IQM — VTT
Q50's hardware partner — ships ``qiskit-iqm`` as the documented path from a local circuit to
that device, so the eventual hardware step is a provider swap rather than a rewrite; and
because Aer gives both an exact statevector simulator and a shot-based, optionally
noise-modelled one behind a single package.

The import gate
---------------
⚠ ``qiskit`` and ``qiskit_aer`` are imported **inside the methods that need them**, through
:func:`kuiva.qc.gate.require`, never at module scope. ``tests/test_qc_skeleton.py`` asserts
this from the sources, because the failure mode is invisible on the machine that introduces
it: an eager import at the top of this file works perfectly there and breaks ``import kuiva``
everywhere else. The stack is built by ``scripts/bootstrap/90_qiskit.sh`` into
``external/venv_qc``, **a different interpreter** from the pinned-version baseline (Qiskit dropped 3.9;
``qiskit-iqm`` requires >= 3.10), so this adapter runs only under that interpreter with the
repository root on ``PYTHONPATH``.

Conventions that had to be checked rather than assumed
-------------------------------------------------------
* **Statevector index.** Qiskit is little-endian in qubit number, the same as
  :mod:`kuiva.qc.circuits` and as a determinant mask, so amplitudes need no permutation and a
  measured bitstring is an occupation mask under ``int(bits, 2)``.
* ⚠ **Pauli labels are the reverse of Kuiva's.** ``Pauli("ZY")`` means qubit 1 is ``Z`` and
  qubit 0 is ``Y``; :func:`kuiva.qc.mapping.pauli_label` writes qubit 0 first. The reversal
  happens here and nowhere else, which is why the label function is not simply written in
  Qiskit's order to begin with — the encoding Kuiva owns should read in Kuiva's order, and
  exactly one place should know that a vendor disagrees.
* **Global phase.** Kuiva's gate matrices are the standard ``exp(-i theta a / 2)``
  definitions, chosen to match Qiskit's, so the two statevectors agree including phase. The
  stub-versus-Aer test nevertheless compares up to a global phase, which is the physically
  meaningful invariant and immune to a transpiler that inserts one.

What ``estimate`` does, stated plainly
---------------------------------------
* ``shots=None``: the exact expectation value, from Aer's statevector through Qiskit's own
  ``Statevector.expectation_value``. Deliberately Qiskit's implementation rather than
  Kuiva's — an independent evaluation of the same quantity is worth more here than a shared
  one, since a phase-convention error in :mod:`kuiva.qc.mapping` is exactly what this
  can fail on.
* ``shots`` given: a real measurement — the qubits are rotated into a measurement basis
  (``H`` for ``X``, ``S^dag H`` for ``Y``) and the parity of the outcome bits is averaged.
  One circuit per **qubit-wise commuting group** (:func:`kuiva.qc.mapping.qwc_groups`), not
  one per Pauli string: strings agreeing wherever they overlap share a basis and therefore
  share their shots. ⚠ Nothing is resampled classically — a shot-based value here is a
  measurement, and an exact value looking like one is the thing this adapter must never
  produce.

References
----------------------------
* A. Javadi-Abhari et al., "Quantum computing with Qiskit", arXiv:2405.08810 (2024).
* Qiskit Aer, github.com/Qiskit/qiskit-aer — the simulator this adapter drives.
"""
from __future__ import annotations

from typing import FrozenSet, Optional

import numpy as np

from ...ci.strings import popcount
from ...util.logging import get_logger
from ..backend import BackendProvenance, EstimateResult, SampleResult
from ..circuits import CircuitSpec
from ..mapping import pauli_label, qwc_groups_from_masks

log = get_logger(__name__)

#: Qiskit method names for the two single-qubit rotations Kuiva names the same way; every
#: other gate in the vocabulary shares its name with Qiskit's method, so the translation is
#: ``getattr`` and this table stays empty until a vendor disagrees.
_GATE_ALIASES = {}


class QiskitAerBackend:
    """Kuiva's :class:`~kuiva.qc.backend.QuantumBackend` over Qiskit Aer.

    Parameters
    ----------
    method : str
        Aer simulation method (``"statevector"``, ``"density_matrix"``, ``"matrix_product_
        state"``, ...). ⚠ Only ``"statevector"`` supports the :meth:`statevector` primitive;
        another method still samples and estimates, and :meth:`capabilities` reports that
        honestly rather than raising later.
    noise_model : object, optional
        A ``qiskit_aer.noise.NoiseModel``. Passed through untouched; its ``repr`` goes into
        the provenance, because a shot count without a noise model is not a description of a
        result. ⚠ A framework object *into the constructor* is fine — the boundary rule is
        about protocol signatures and results, and nothing here leaks outward.
    optimization_level : int
        Passed to ``transpile``; recorded in the provenance.
    seed : int, optional
        Default simulator seed. A per-call ``seed=`` overrides it, and ``None`` is recorded as
        ``None`` rather than as a number the run did not use.
    device : str
        Provenance label. It is ``"aer_simulator"`` here and would be the device id once a
        provider adapter replaces the simulator.
    """

    name = "qiskit_aer"

    def __init__(self, *, method: str = "statevector", noise_model=None,
                 optimization_level: int = 1, seed: Optional[int] = None,
                 device: str = "aer_simulator") -> None:
        self._method = str(method)
        self._noise_model = noise_model
        self._optimization_level = int(optimization_level)
        self._seed = None if seed is None else int(seed)
        self._device = str(device)
        self._simulator = None
        # Constructing the simulator eagerly is what makes an absent framework a
        # construction-time refusal naming the bootstrap script, rather than a failure in the
        # middle of a macro-iteration.
        self._aer()

    # -- the gate --

    @staticmethod
    def _qiskit():
        from ..gate import require
        return require("qiskit", purpose="the Qiskit Aer backend adapter")

    @staticmethod
    def _quantum_info():
        # ⚠ Imported by name, not reached as ``qiskit.quantum_info``: ``import qiskit`` does
        # not necessarily bind its submodules, and an attribute lookup that happens to work
        # today is a dependency on Qiskit's own import order.
        from ..gate import require
        return require("qiskit.quantum_info", purpose="exact Pauli expectation values")

    @staticmethod
    def _aer_module():
        from ..gate import require
        return require("qiskit_aer", purpose="the Qiskit Aer backend adapter")

    def _aer(self):
        """The ``AerSimulator``, built once and reused."""
        if self._simulator is None:
            aer = self._aer_module()
            kwargs = {"method": self._method}
            if self._noise_model is not None:
                kwargs["noise_model"] = self._noise_model
            self._simulator = aer.AerSimulator(**kwargs)
        return self._simulator

    @property
    def version(self) -> str:
        qiskit = self._qiskit()
        return "qiskit {} / aer {}".format(getattr(qiskit, "__version__", "?"),
                                           getattr(self._aer_module(), "__version__", "?"))

    def capabilities(self) -> FrozenSet[str]:
        """``sample`` and ``estimate`` always; ``statevector`` only for that method.

        Declared from configuration rather than inferred from which methods exist, exactly as
        the protocol requires: a method that raises is a capability claim that fails at the
        worst possible moment.
        """
        primitives = {"sample", "estimate"}
        if self._method == "statevector":
            primitives.add("statevector")
        return frozenset(primitives)

    # -- translation --

    def to_qiskit(self, circuit: CircuitSpec):
        """Translate a :class:`CircuitSpec` into a ``qiskit.QuantumCircuit``.

        Public because it is the useful thing to look at when a disagreement between this
        backend and the stub has to be bisected — and because a translation nobody can inspect
        is where a convention error hides.
        """
        qiskit = self._qiskit()
        qc = qiskit.QuantumCircuit(circuit.n_qubit)
        for gate in circuit.gates:
            method = getattr(qc, _GATE_ALIASES.get(gate.name, gate.name), None)
            if method is None:                                       # pragma: no cover
                raise NotImplementedError(
                    "the Qiskit adapter has no translation for gate {!r}; add it here rather "
                    "than removing the gate from kuiva.qc.circuits.GATE_ARITY, which is the "
                    "algorithm layer's vocabulary and not a device's".format(gate.name))
            method(*gate.params, *gate.qubits)
        return qc

    def _transpiled(self, qc):
        qiskit = self._qiskit()
        return qiskit.transpile(qc, self._aer(),
                                optimization_level=self._optimization_level)

    def _provenance(self, *, shots: Optional[int], seed: Optional[int]) -> BackendProvenance:
        return BackendProvenance(
            backend=self.name, version=self.version, device=self._device,
            shots=shots, seed=seed,
            noise_model="none" if self._noise_model is None else repr(self._noise_model),
            transpilation="qiskit.transpile(optimization_level={}) for AerSimulator("
                          "method={!r})".format(self._optimization_level, self._method))

    def _effective_seed(self, seed: Optional[int]) -> Optional[int]:
        return self._seed if seed is None else int(seed)

    # -- the primitives --

    def statevector(self, circuit: CircuitSpec) -> np.ndarray:
        """The amplitudes, little-endian in qubit number (Qiskit's convention and Kuiva's)."""
        if "statevector" not in self.capabilities():
            raise NotImplementedError(
                "this backend was constructed with method={!r}, which does not provide the "
                "statevector primitive; construct it with method='statevector'"
                .format(self._method))
        qc = self.to_qiskit(circuit)
        qc.save_statevector()
        result = self._aer().run(self._transpiled(qc)).result()
        return np.ascontiguousarray(np.asarray(result.get_statevector(), dtype=complex),
                                    dtype=np.complex128)

    def sample(self, circuit: CircuitSpec, shots: int, *,
               seed: Optional[int] = None) -> SampleResult:
        """Measure every qubit in the computational basis ``shots`` times."""
        shots = int(shots)
        if shots <= 0:
            raise ValueError("shots must be positive, got {}".format(shots))
        qc = self.to_qiskit(circuit)
        qc.measure_all()
        effective = self._effective_seed(seed)
        result = self._aer().run(self._transpiled(qc), shots=shots,
                                 seed_simulator=effective).result()
        counts = result.get_counts()
        # Qiskit's count keys are bitstrings with qubit 0 rightmost, so int(key, 2) is the
        # occupation mask directly — the same integer a determinant carries.
        masks = np.array([int(key.replace(" ", ""), 2) for key in counts], dtype=np.uint64)
        values = np.array([counts[key] for key in counts], dtype=np.int64)
        order = np.argsort(masks)
        return SampleResult(masks=masks[order], counts=values[order],
                            provenance=self._provenance(shots=shots, seed=effective))

    def estimate(self, circuit: CircuitSpec, x_masks: np.ndarray, z_masks: np.ndarray, *,
                 shots: Optional[int] = None,
                 seed: Optional[int] = None) -> EstimateResult:
        """Pauli expectation values. See the module docstring for what each mode really does."""
        x = np.ascontiguousarray(x_masks, dtype=np.uint64)
        z = np.ascontiguousarray(z_masks, dtype=np.uint64)
        if x.shape != z.shape or x.ndim != 1:
            raise ValueError("x_masks {} and z_masks {} must be matching 1-D arrays"
                             .format(x.shape, z.shape))
        if shots is None:
            return self._estimate_exact(circuit, x, z)
        return self._estimate_sampled(circuit, x, z, int(shots), seed)

    def _estimate_exact(self, circuit: CircuitSpec, x, z) -> EstimateResult:
        info = self._quantum_info()
        psi = info.Statevector(self.statevector(circuit), dims=(2,) * circuit.n_qubit)
        values = np.empty(x.size, dtype=np.float64)
        for t in range(x.size):
            # Reversed: Qiskit writes the highest qubit first, Kuiva the lowest.
            label = pauli_label(int(x[t]), int(z[t]), circuit.n_qubit)[::-1]
            values[t] = np.real(psi.expectation_value(info.Pauli(label)))
        return EstimateResult(values, np.zeros_like(values),
                              self._provenance(shots=None, seed=None))

    def _estimate_sampled(self, circuit: CircuitSpec, x, z, shots: int,
                          seed: Optional[int]) -> EstimateResult:
        """One circuit per **qubit-wise commuting group**, not one per Pauli string.

        Strings that carry the same single-qubit Pauli wherever they overlap are measured
        together: rotate each qubit into the group's basis once, take the shots once, and read
        every member's parity off the same bitstrings. The reduction is measured rather than
        assumed — ``tests/test_qc_mapping.py`` reports it, and it is a factor of 2.5-3.5 on the
        Hamiltonians reached so far, growing slowly with the active space.

        ⚠ **Grouping changes the shot noise's correlation structure, not its bias.** Every
        value here is still an unbiased estimate with the same per-term variance, but terms in
        one group are now *correlated*, which is precisely the independence assumption
        :meth:`~kuiva.qc.backend.EstimateResult.combine` documents as the caller's to justify.
        The group structure is therefore recorded on the result. ⚠ It also means this backend
        and the stub do **not** produce statistically identical shot records for the same seed:
        both are unbiased, their correlations differ, and a test comparing them must compare
        expectation values within the standard error rather than draw for draw.
        """
        if shots <= 0:
            raise ValueError("shots must be positive, got {}".format(shots))
        qiskit = self._qiskit()
        effective = self._effective_seed(seed)
        base = self.to_qiskit(circuit)
        groups = qwc_groups_from_masks(x, z)
        # The identity string is exactly 1 with zero variance — no circuit, and no shot spent
        # measuring a constant. Everything else is overwritten below.
        values = np.ones(x.size, dtype=np.float64)
        variances = np.zeros(x.size, dtype=np.float64)
        circuits, live = [], []
        for members in groups:
            basis_x = int(np.bitwise_or.reduce(x[members]))
            basis_z = int(np.bitwise_or.reduce(z[members]))
            support = [k for k in range(circuit.n_qubit)
                       if (basis_x >> k) & 1 or (basis_z >> k) & 1]
            if not support:                       # a group holding only the identity string
                continue
            qc = base.copy()
            for k in support:
                has_x, has_z = (basis_x >> k) & 1, (basis_z >> k) & 1
                if has_x and has_z:               # Y: S^dag then H maps Y -> Z
                    qc.sdg(k)
                    qc.h(k)
                elif has_x:                       # X: H maps X -> Z
                    qc.h(k)
            creg = qiskit.ClassicalRegister(len(support))
            qc.add_register(creg)
            qc.measure(support, list(range(len(support))))
            circuits.append(qc)
            live.append((members, support))
        if circuits:
            result = self._aer().run(self._transpiled(circuits), shots=shots,
                                     seed_simulator=effective).result()
            for slot, (members, support) in enumerate(live):
                counts = result.get_counts(slot)
                # Classical bit j holds qubit support[j], and Qiskit writes bit 0 rightmost, so
                # int(key, 2) is a mask over the classical register.
                outcomes = np.array([int(key.replace(" ", ""), 2) for key in counts],
                                    dtype=np.uint64)
                weights = np.array([counts[key] for key in counts], dtype=np.float64)
                total = float(weights.sum())
                position = {q: j for j, q in enumerate(support)}
                for t in members.tolist():
                    term = 0
                    for k in range(circuit.n_qubit):
                        if (int(x[t]) >> k) & 1 or (int(z[t]) >> k) & 1:
                            term |= 1 << position[k]
                    parity = 1.0 - 2.0 * (popcount(outcomes & np.uint64(term)) & 1)
                    values[t] = float(weights @ parity) / total
                    variances[t] = max(0.0, 1.0 - values[t] ** 2) / total
        return EstimateResult(values, variances,
                              self._provenance(shots=shots, seed=effective),
                              groups=groups)

    def __repr__(self) -> str:
        return "QiskitAerBackend(method={!r}, noise={}, device={!r})".format(
            self._method, "none" if self._noise_model is None else "set", self._device)


__all__ = ["QiskitAerBackend"]
