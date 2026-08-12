"""Kuiva's own exact statevector simulator: the declared stub implementation of the boundary.

**Orchestration, not a registered kernel.**

Why "stub" and what that word does *not* mean here
---------------------------------------------------
It names a **role**, not a quality. ``amf/backend.py`` keeps a stub second backend because an
interface with one implementation is indistinguishable from no interface, and
``tests/test_amf_backend.py`` is forbidden from deleting it for looking trivial. The same rule
applies to this boundary, twice over: this backend is also what keeps the
*entire* algorithm layer testable in the default ``pytest`` run with ``external/venv_qc``
absent. Everything it computes is exact — the amplitudes are the amplitudes, the expectation
values are the expectation values, the shot noise is drawn from the true distribution.

⚠ **It is a declared implementation, never a fallback.** :mod:`kuiva.qc.gate` refuses a
missing framework rather than quietly substituting this (the refuse-never-reconcile culture of the method surface); nothing
anywhere selects this backend because another one failed.

``sample`` on top of ``statevector`` is the legitimate, *declared* emulation
----------------------------------------------------------------------------
The backend design permits exactly this and requires it be explicit: probabilities come from
``|psi|^2`` and shots from a seeded multinomial, which is the correct sampling distribution
for a noiseless device. The provenance records ``device="statevector"`` and the seed, so a run
is replayable — the promise the provenance rule makes for simulator results.

⚠ **What it deliberately does not model:** noise of any kind, connectivity, native gate sets,
readout error, or a queue. A number from here is what a *perfect* device would give, and no
statement about hardware feasibility may be made from it. That is Aer's job (with a noise
model) and, eventually, the device's.

Cost, and the refusal that goes with it
----------------------------------------
``2^n`` amplitudes, checked against the configured memory limit before the allocation. Twenty-something
qubits is the practical wall on a laptop and the refusal says so rather than swapping.
"""
from __future__ import annotations

from typing import FrozenSet, Optional

import numpy as np

from ...util import resources as res
from ...util.logging import get_logger
from ..backend import BackendProvenance, EstimateResult, SampleResult
from ..circuits import CircuitSpec, apply_circuit
from ..mapping import pauli_expectation

log = get_logger(__name__)

#: Version of *this* implementation, for provenance. Bumped when the numerical content of a
#: result changes for an unchanged request — the same discipline the AMF cache's formula
#: version carries, because a recorded provenance outlives the code that wrote it.
STUB_VERSION = "1"


def statevector_gb(n_qubit: int) -> float:
    """Size [GB] of the ``2^n`` complex amplitudes. Exact, never padded."""
    return res.array_gb((1 << int(n_qubit),), np.complex128)


class StubBackend:
    """Exact simulation of the :class:`~kuiva.qc.circuits.CircuitSpec` vocabulary.

    Parameters
    ----------
    seed : int, optional
        Default seed for :meth:`sample` and for shot noise in :meth:`estimate`. A per-call
        ``seed=`` overrides it. ⚠ Recorded in the provenance either way, and ``None`` is
        recorded as ``None``: a provenance claiming a seed the run did not use is worse than
        one admitting it is not replayable.
    """

    name = "stub"

    def __init__(self, *, seed: Optional[int] = None) -> None:
        self._seed = None if seed is None else int(seed)

    @property
    def version(self) -> str:
        return STUB_VERSION

    def capabilities(self) -> FrozenSet[str]:
        """All three primitives — ``sample`` by the declared emulation described above."""
        return frozenset(("sample", "estimate", "statevector"))

    # -- provenance --

    def _provenance(self, *, shots: Optional[int], seed: Optional[int]) -> BackendProvenance:
        return BackendProvenance(
            backend=self.name, version=self.version, device="statevector",
            shots=shots, seed=seed, noise_model="none",
            transpilation="none (the CircuitSpec vocabulary is executed directly)")

    def _rng(self, seed: Optional[int]):
        effective = self._seed if seed is None else int(seed)
        return np.random.default_rng(effective), effective

    # -- the primitives --

    def statevector(self, circuit: CircuitSpec) -> np.ndarray:
        """Amplitudes of ``circuit`` applied to ``|0...0>``, little-endian in qubit number."""
        res.require("statevector simulation", statevector_gb(circuit.n_qubit),
                    note="{} qubits, {} gates".format(circuit.n_qubit, circuit.n_gates),
                    advice=["a smaller active space",
                            "a sampling algorithm (kuiva.qc's sampled-subspace driver) never "
                            "forms the statevector"])
        psi = np.zeros(1 << circuit.n_qubit, dtype=np.complex128)
        psi[0] = 1.0
        return apply_circuit(circuit, psi)

    def sample(self, circuit: CircuitSpec, shots: int, *,
               seed: Optional[int] = None) -> SampleResult:
        """Computational-basis shots, drawn from the exact ``|psi|^2``."""
        shots = int(shots)
        if shots <= 0:
            raise ValueError("shots must be positive, got {}".format(shots))
        psi = self.statevector(circuit)
        probabilities = np.abs(psi) ** 2
        total = probabilities.sum()
        # Renormalize against rounding only: the circuit is unitary, so anything else here
        # would be hiding a defect rather than a floating-point residue.
        if not np.isclose(total, 1.0, rtol=0.0, atol=1e-10):
            raise ValueError("the simulated state has norm {:.12f}, not 1; a gate in the "
                             "vocabulary is not unitary".format(float(total)))
        rng, effective_seed = self._rng(seed)
        counts = rng.multinomial(shots, probabilities / total)
        hit = np.nonzero(counts)[0]
        return SampleResult(masks=hit.astype(np.uint64), counts=counts[hit].astype(np.int64),
                            provenance=self._provenance(shots=shots, seed=effective_seed))

    def estimate(self, circuit: CircuitSpec, x_masks: np.ndarray, z_masks: np.ndarray, *,
                 shots: Optional[int] = None,
                 seed: Optional[int] = None) -> EstimateResult:
        """Pauli expectation values; exact when ``shots is None``.

        With ``shots`` given, each estimate is drawn from the **exact** distribution of a
        shot-based measurement of that Pauli: a Pauli string has eigenvalues ``+-1``, so the
        number of ``+1`` outcomes is ``Binomial(shots, (1 + <P>)/2)`` and the estimator is
        ``2 k / shots - 1``. That is the true sampling distribution, not a Gaussian
        approximation of it, and the returned variance is the variance **of the estimator**.
        """
        psi = self.statevector(circuit)
        exact = pauli_expectation(x_masks, z_masks, psi)
        if shots is None:
            return EstimateResult(exact, np.zeros_like(exact),
                                  self._provenance(shots=None, seed=None))
        shots = int(shots)
        if shots <= 0:
            raise ValueError("shots must be positive, got {}".format(shots))
        rng, effective_seed = self._rng(seed)
        p_plus = np.clip(0.5 * (1.0 + exact), 0.0, 1.0)
        values = 2.0 * rng.binomial(shots, p_plus) / shots - 1.0
        variances = np.clip(1.0 - values ** 2, 0.0, None) / shots
        return EstimateResult(values, variances,
                              self._provenance(shots=shots, seed=effective_seed))

    def __repr__(self) -> str:
        return "StubBackend(version={}, seed={})".format(self.version, self._seed)


__all__ = ["STUB_VERSION", "StubBackend", "statevector_gb"]
