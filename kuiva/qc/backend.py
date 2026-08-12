"""The hardware boundary: task primitives, declared capabilities, and the backend registry.

**Orchestration, not a registered kernel.**

What a backend is
-----------------
At most three narrow methods, each returning plain arrays plus a provenance record:

``sample(circuit, shots)``
    Measure the circuit in the computational basis and return ``(masks, counts)``. The masks
    are ``uint64`` occupation bitmasks — literally ``ci/strings.py``'s determinant dtype
, because Jordan-Wigner makes the computational basis and the determinant basis the
    same object. This is the only primitive a sampled-subspace algorithm (SQD, SKQD) needs.
``estimate(circuit, x_masks, z_masks)``
    Expectation values and variances of Pauli strings given in :mod:`kuiva.qc.mapping`'s
    symplectic encoding. What a VQE-class algorithm needs.
``statevector(circuit)``
    The amplitudes. Simulators only; the zero-noise, Tier-0-style validation mode.

A backend **declares** which of these it supports, and an algorithm **declares** which it
requires. The pairing is validated once, at construction, by :func:`require_primitives`, and
an unsupported pairing is refused with a message naming both sides — never silently emulated
or degraded (the refuse-never-reconcile culture of the method surface, restated for this boundary). Emulating ``sample``
on top of ``statevector`` is legitimate when it is *declared*: the stub backend does exactly
that, as its implementation, not as a fallback.

The sampler/estimator split is not an invention. It is the shape the whole field converged on
— Qiskit primitives, and every vendor cloud API underneath — so an adapter for a new stack is
a translation exercise rather than a redesign.

The four rules inherited verbatim from ``amf/backend.py``
----------------------------------------------------------------------
1. **Plain NumPy arrays and metadata only** across the boundary, both directions. No framework
   object appears in any protocol signature or in any result; :class:`~kuiva.qc.circuits
   .CircuitSpec` makes this concrete for circuits and the symplectic encoding makes it
   concrete for operators.
2. **A stub implementation exists from day one and is never deleted** once a real adapter
   arrives. ``tests/test_amf_backend.py`` keeps its stub for the reason that applies twice
   over here: an interface with one implementation is indistinguishable from no interface, and
   Kuiva's stub is additionally what keeps the *entire* algorithm layer testable in the
   default suite with ``external/venv_qc`` absent.
3. **Real adapters are import-gated**, framework imports inside the function that needs them,
   behind :func:`kuiva.qc.gate.require`, never at package ``__init__``. They register in the
   name-to-factory table below, so a Cirq, cuQuantum or future-IQM adapter is an entry in the
   same table, selected by name.
4. **The dependency runs one way.** Backend results are consumed inside :mod:`kuiva.qc` and
   turned into the plain ``(energy, gamma, Gamma)`` / ``AdaptiveCISolver`` contract there.
   Nothing downstream of ``mcscf/orbopt.py`` ever sees a qubit, a circuit or a shot count.

Provenance is part of the result, not an afterthought
------------------------------------------------------
:class:`BackendProvenance` travels with every result — backend and version, device, shots,
seed, noise model, transpilation summary — exactly as ``ScreeningRecord`` and
``DecouplingRecord`` do. An energy whose record does not say which device and how
many shots produced it is not interpretable, and the record outlives the session. ⚠ A
simulator result must be **replayable from its recorded seed**; a backend that cannot promise
that must record ``seed=None`` rather than a number it did not use. If a QC-solved CASSCF ever
reaches the property dump, this record joins ``provenance()`` in the header.

Two things the protocol may never assume
-----------------------------------------
* ⚠ **That calls are fast.** Real devices are reached through queued cloud jobs with
  minutes-to-hours latency. The protocol stays synchronous from the caller's view — the event driver's
  ``event_interval`` is already the right control for an expensive proposal, and the checkpoint policy's
  checkpointing means a macro-iteration blocked on a queue can be killed and resumed — but an
  adapter is free to implement ``sample`` over an asynchronous job API.
* ⚠ **That the adapter is in this process.** Only plain arrays and strings cross, so an
  adapter may run in a different process or interpreter with the arrays serialized across.
  Not built now; the binding rule is that nothing here may acquire an in-process assumption —
  no callback from a backend into Kuiva, no shared mutable state, no open file handle in a
  request or a result.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, Optional, Sequence, Tuple

import numpy as np

try: # Protocol is 3.8+; the project targets 3.9
    from typing import Protocol, runtime_checkable
except ImportError:                      # pragma: no cover - defensive, never taken on 3.9
    Protocol = object                    # type: ignore

    def runtime_checkable(cls):          # type: ignore
        return cls

from ..util.logging import get_logger

log = get_logger(__name__)

#: The task primitives a backend may declare. Nothing else is negotiable at this boundary; a
#: capability that is really an *option* (a noise model, a device name) belongs in the
#: adapter's constructor and in its provenance, not here.
PRIMITIVES: Tuple[str, ...] = ("sample", "estimate", "statevector")


class CapabilityError(TypeError):
    """An algorithm was paired with a backend that cannot run it.

    A :class:`TypeError` because it is a statement about interfaces rather than about values,
    and it is raised at construction — never mid-calculation, and never in place of doing the
    thing some other way.
    """


# --- Provenance ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BackendProvenance:
    """What produced a backend result. ⚠ A contract with stored data: add fields, do not
    rename or repurpose them (the rule ``ScreeningRecord`` carries)."""

    backend: str = ""
    version: str = ""
    device: str = ""
    shots: Optional[int] = None
    seed: Optional[int] = None
    noise_model: str = "none"
    transpilation: str = ""
    extra: Dict[str, str] = field(default_factory=dict)

    @property
    def exact(self) -> bool:
        """Whether the result carries no sampling noise (``shots is None``) — which is a
        property of *this* result, not of the backend."""
        return self.shots is None

    def as_dict(self) -> Dict[str, object]:
        """JSON-serializable, plain builtins only, so it round-trips unchanged."""
        return {
            "backend": str(self.backend),
            "version": str(self.version),
            "device": str(self.device),
            "shots": None if self.shots is None else int(self.shots),
            "seed": None if self.seed is None else int(self.seed),
            "noise_model": str(self.noise_model),
            "transpilation": str(self.transpilation),
            "extra": {str(k): str(v) for k, v in sorted(self.extra.items())},
        }

    def __repr__(self) -> str:
        return "BackendProvenance({} {}, device={!r}, shots={}, seed={}, noise={})".format(
            self.backend, self.version, self.device, self.shots, self.seed, self.noise_model)


# --- Results ------------------------------------------------------------------------------

@dataclass(frozen=True)
class SampleResult:
    """Computational-basis measurement outcomes: unique masks and their counts.

    ``masks`` is ``uint64`` and ascending; ``counts`` is ``int64`` and strictly positive.
    Unobserved outcomes are absent rather than zero — the whole point of a sampled subspace is
    that it is small compared with the space it was drawn from.
    """

    masks: np.ndarray
    counts: np.ndarray
    provenance: BackendProvenance

    def __post_init__(self) -> None:
        object.__setattr__(self, "masks", np.ascontiguousarray(self.masks, dtype=np.uint64))
        object.__setattr__(self, "counts", np.ascontiguousarray(self.counts, dtype=np.int64))
        if self.masks.shape != self.counts.shape or self.masks.ndim != 1:
            raise ValueError("masks {} and counts {} must be matching 1-D arrays".format(
                self.masks.shape, self.counts.shape))
        # Comparison, not np.diff: differencing uint64 wraps, so a descending pair would come
        # back as a huge positive and pass.
        if np.any(self.counts <= 0) or np.any(self.masks[1:] <= self.masks[:-1]):
            raise ValueError("SampleResult holds unique ascending masks with positive counts")

    @property
    def shots(self) -> int:
        return int(self.counts.sum())

    @property
    def n_unique(self) -> int:
        return int(self.masks.size)

    def probabilities(self) -> np.ndarray:
        return self.counts.astype(np.float64) / max(1, self.shots)

    def __repr__(self) -> str:
        return "SampleResult(n_unique={}, shots={}, backend={})".format(
            self.n_unique, self.shots, self.provenance.backend)


@dataclass(frozen=True)
class EstimateResult:
    """Pauli expectation values and the variance of each estimate.

    ⚠ ``variances`` is the variance **of the estimator**, i.e. it already carries the ``1 /
    shots``; it is zero for an exact (shot-free) evaluation. Reporting the observable's
    variance instead would differ by orders of magnitude and look entirely plausible.
    """

    values: np.ndarray
    variances: np.ndarray
    provenance: BackendProvenance
    groups: Optional[Tuple[np.ndarray, ...]] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", np.ascontiguousarray(self.values, dtype=np.float64))
        object.__setattr__(self, "variances",
                           np.ascontiguousarray(self.variances, dtype=np.float64))
        if self.values.shape != self.variances.shape or self.values.ndim != 1:
            raise ValueError("values {} and variances {} must be matching 1-D arrays".format(
                self.values.shape, self.variances.shape))
        if self.groups is not None:
            object.__setattr__(self, "groups",
                               tuple(np.ascontiguousarray(g, dtype=np.int64)
                                     for g in self.groups))

    @property
    def n_circuits(self) -> int:
        """Circuit executions this result cost — the honest measurement-cost figure.

        One per commuting group where the backend grouped, one per term where it did not.
        Exact evaluations cost no circuits in the measurement sense and report ``0``.
        """
        if self.provenance.exact:
            return 0
        return self.values.size if self.groups is None else len(self.groups)

    def combine(self, coeffs: np.ndarray) -> Tuple[float, float]:
        """``(sum_t c_t <P_t>, sum_t c_t^2 var_t)`` — the energy of a :class:`PauliSum` and
        the variance of that estimate, assuming the terms were estimated independently.

        ⚠ **The independence assumption is the caller's to justify, and** :attr:`groups`
        **records where it fails.** It holds for one circuit execution per term; once
        qubit-wise commuting groups are measured together, the terms within a group are
        correlated — that covariance is exactly what the grouping buys, and it can raise or
        lower the variance of the *sum*. The value returned is unaffected (every term is still
        unbiased); only the second element of the tuple becomes an approximation, and a caller
        quoting an error bar from a grouped estimate must say so.
        """
        c = np.ascontiguousarray(coeffs, dtype=np.float64)
        if c.shape != self.values.shape:
            raise ValueError("expected {} coefficients, got {}".format(
                self.values.size, c.size))
        return float(c @ self.values), float((c * c) @ self.variances)

    def __repr__(self) -> str:
        return "EstimateResult(n_terms={}, n_circuits={}, backend={})".format(
            self.values.size, self.n_circuits, self.provenance.backend)


# --- The protocol -------------------------------------------------------------------------

@runtime_checkable
class QuantumBackend(Protocol):
    """What a quantum backend must provide. See the module docstring for the rules."""

    #: Registry name, e.g. ``"stub"`` or ``"qiskit_aer"``.
    name: str

    @property
    def version(self) -> str:
        """Version of the underlying implementation, for provenance."""
        ...

    def capabilities(self) -> FrozenSet[str]:
        """The subset of :data:`PRIMITIVES` this backend actually implements.

        ⚠ Declared, not inferred from which methods exist: a method that raises
        ``NotImplementedError`` is a capability claim that fails at the worst moment, which is
        precisely what the negotiation below exists to prevent.
        """
        ...

    def sample(self, circuit, shots: int, *, seed: Optional[int] = None) -> SampleResult:
        """Measure ``circuit`` in the computational basis ``shots`` times."""
        ...

    def estimate(self, circuit, x_masks: np.ndarray, z_masks: np.ndarray, *,
                 shots: Optional[int] = None,
                 seed: Optional[int] = None) -> EstimateResult:
        """Expectation values of the Pauli strings ``(x_masks, z_masks)`` in ``circuit``'s
        state. ``shots=None`` asks for an exact evaluation and is legal only on a simulator."""
        ...

    def statevector(self, circuit) -> np.ndarray:
        """The ``2^n`` complex amplitudes, little-endian in qubit number."""
        ...


def require_primitives(backend, needed: Sequence[str], *, algorithm: str) -> None:
    """Validate an algorithm/backend pairing **once, at construction**.

    Raises :class:`CapabilityError` naming both sides, what was required and what is on offer.
    The message is the feature: the person who hits this is choosing between two names in a
    registry and needs to know which pairings exist, not that something is unsupported.
    """
    needed = tuple(str(p) for p in needed)
    unknown = [p for p in needed if p not in PRIMITIVES]
    if unknown:
        raise ValueError("algorithm {!r} requires unknown primitive(s) {}; the boundary "
                         "offers {}".format(algorithm, unknown, list(PRIMITIVES)))
    have = frozenset(backend.capabilities())
    missing = [p for p in needed if p not in have]
    if missing:
        raise CapabilityError(
            "algorithm {!r} requires the primitive(s) {} but backend {!r} declares only {}. "
            "This pairing is refused rather than emulated: pick a backend that provides them "
            "(kuiva.qc.backend.available_backends()) or an algorithm that does not need them."
            .format(algorithm, ", ".join(missing), getattr(backend, "name", backend),
                    ", ".join(sorted(have)) or "no primitives"))


# --- The registry -------------------------------------------------------------------------

_BACKENDS: Dict[str, Callable[..., QuantumBackend]] = {}


def register_backend(name: str, factory: Callable[..., QuantumBackend]) -> None:
    """Register a backend factory under ``name``.

    ``factory`` is called lazily, per lookup, and receives whatever keyword arguments
    :func:`get_backend` was given. Lazy because an adapter's import may be expensive or may
    fail where its framework is absent, and a lookup for a *different* backend must not care.
    """
    key = str(name).lower()
    if key in _BACKENDS:
        log.debug("replacing already-registered quantum backend %r", key)
    _BACKENDS[key] = factory


def unregister_backend(name: str) -> None:
    """Remove a backend (tests)."""
    _BACKENDS.pop(str(name).lower(), None)


def available_backends() -> Tuple[str, ...]:
    """Every registered name. ⚠ Registration says a backend is *known*, not that its
    framework is installed — that is only discoverable by constructing it, which is where
    :func:`kuiva.qc.gate.require` raises with the script to run."""
    return tuple(sorted(_BACKENDS))


def get_backend(name: str = "stub", **kwargs) -> QuantumBackend:
    """Instantiate a registered backend by name."""
    key = str(name).lower()
    if key not in _BACKENDS:
        raise ValueError("unknown quantum backend {!r}; registered: {}".format(
            name, ", ".join(available_backends()) or "(none)"))
    return _BACKENDS[key](**kwargs)


def _register_default_backends() -> None:
    """Register the backends that ship with Kuiva.

    Both factories import their module lazily, so ``import kuiva.qc.backend`` pulls in neither
    a simulator nor a framework — which is what ``tests/test_qc_skeleton.py`` asserts from the
    sources.
    """
    def _stub(**kwargs) -> QuantumBackend:
        from .backends.stub import StubBackend
        return StubBackend(**kwargs)

    def _qiskit_aer(**kwargs) -> QuantumBackend:
        from .backends.qiskit_aer import QiskitAerBackend
        return QiskitAerBackend(**kwargs)

    register_backend("stub", _stub)
    register_backend("qiskit_aer", _qiskit_aer)


_register_default_backends()


__all__ = ["PRIMITIVES", "BackendProvenance", "CapabilityError", "EstimateResult",
           "QuantumBackend", "SampleResult", "available_backends", "get_backend",
           "register_backend", "require_primitives", "unregister_backend"]
