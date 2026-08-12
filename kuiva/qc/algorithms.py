"""The algorithm registry: the second axis of :mod:`kuiva.qc`.

**Orchestration, not a registered kernel.**

Why a registry rather than a constructor call
-----------------------------------------------
Quantum CI algorithms are appearing faster than the hardware is — sample-based Krylov
diagonalization arrived within a year of SQD itself — so *which algorithm* has to be as
swappable as *which device*. This is the third instance of the project's one extension
pattern (``ci/kernels.py``, ``amf/backend.py``, ``qc/backend.py``), and the rules are the same
ones the Hamiltonian method surface imposes on decoupling x screening:

* **Algorithm and backend are two orthogonal name-to-factory registries.** Any algorithm runs
  on any backend that declares the primitives it requires.
* **An incapable pairing is refused at construction, naming both sides** — never emulated,
  never degraded. :func:`build_ci_solver` does that check *before* constructing anything, so
  the message arrives while the caller is still choosing names rather than half way through a
  macro-iteration.
* **Adding an algorithm touches only** :mod:`kuiva.qc`: implement the adaptive-solver protocol (kuiva/mcscf/adaptive.py),
  register the name, declare the capabilities. Nothing in ``mcscf/`` — and nothing in any
  backend adapter — learns that a new algorithm exists.
* ⚠ **Only implemented algorithms are registered**, as ``amf/backend.py`` keeps ``"kuiva"``
  absent: a name that resolves to something non-functional, or worse to a *different*
  algorithm, fails further from its cause than a name that does not resolve.
* ⚠ **A name that is a configuration of another algorithm says so.** ``"skqd"`` is
  ``"sqd"`` with :class:`~kuiva.qc.ansatz.TimeEvolutionStrategy` as its circuit source and
  *nothing else*, because that is what sample-based Krylov diagonalization is — everything
  downstream of the sampling step is identical. It gets its own registry
  entry because that is the name a caller will look for, and it **refuses a ``strategy=``
  argument** rather than letting the name describe a run it did not perform.

What every registered algorithm must satisfy
----------------------------------------------
The :class:`~kuiva.mcscf.adaptive.AdaptiveCISolver` protocol of the adaptive layer — or the plain
``ci_solver(ints) -> (energy, gamma, Gamma)`` contract, which
``mcscf.adaptive.as_adaptive_solver`` wraps. A sampling solver is an adaptive one by
construction, so in practice everything here implements the four-method protocol.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

from ..util.logging import get_logger
from .backend import PRIMITIVES, get_backend, require_primitives

log = get_logger(__name__)


@dataclass(frozen=True)
class AlgorithmSpec:
    """One registered algorithm: how to build it, and what it needs of a backend.

    ``requires`` is declared here as well as inside the solver so that
    :func:`build_ci_solver` can refuse a pairing **without constructing anything** — which
    matters once a solver's constructor does real work, and which is what lets a caller
    enumerate the legal pairings up front.
    """

    name: str
    factory: Callable[..., Any]
    requires: Tuple[str, ...]
    description: str = ""

    def __post_init__(self) -> None:
        unknown = [p for p in self.requires if p not in PRIMITIVES]
        if unknown:
            raise ValueError("algorithm {!r} declares unknown primitive(s) {}; the boundary "
                             "offers {}".format(self.name, unknown, list(PRIMITIVES)))

    def __repr__(self) -> str:
        return "AlgorithmSpec({}, requires={})".format(self.name, list(self.requires))


_ALGORITHMS: Dict[str, AlgorithmSpec] = {}


def register_algorithm(name: str, factory: Callable[..., Any], requires, *,
                       description: str = "") -> None:
    """Register a CI algorithm under ``name``."""
    key = str(name).lower()
    if key in _ALGORITHMS:
        log.debug("replacing already-registered quantum CI algorithm %r", key)
    _ALGORITHMS[key] = AlgorithmSpec(name=key, factory=factory,
                                     requires=tuple(str(p) for p in requires),
                                     description=description)


def unregister_algorithm(name: str) -> None:
    """Remove an algorithm (tests)."""
    _ALGORITHMS.pop(str(name).lower(), None)


def available_algorithms() -> Tuple[str, ...]:
    return tuple(sorted(_ALGORITHMS))


def algorithm_spec(name: str) -> AlgorithmSpec:
    """The :class:`AlgorithmSpec` registered under ``name``; refuse, naming what exists."""
    key = str(name).lower()
    if key not in _ALGORITHMS:
        raise ValueError(
            "unknown quantum CI algorithm {!r}; registered: {}.".format(
                name, ", ".join(available_algorithms()) or "(none)"))
    return _ALGORITHMS[key]


def resolve_ci_solver(name: str = "sqd") -> Callable[..., Any]:
    """The factory registered under ``name``.

    ⚠ Does **no** capability checking, because it has no backend to check against. Use
    :func:`build_ci_solver` unless you are constructing the backend yourself — in which case
    the solver's own constructor performs the same check.
    """
    return algorithm_spec(name).factory


def build_ci_solver(name: str, *, backend: Any = "stub", **kwargs):
    """Construct algorithm ``name`` on ``backend``, refusing an incapable pairing first.

    ``backend`` is a registered name or an already-constructed backend. The capability check
    runs **before** the factory, so an incapable pairing costs nothing and the message names
    both sides.
    """
    spec = algorithm_spec(name)
    instance = get_backend(backend) if isinstance(backend, str) else backend
    require_primitives(instance, spec.requires, algorithm=spec.name)
    return spec.factory(backend=instance, **kwargs)


def _register_default_algorithms() -> None:
    """Register what is implemented. See the module docstring on what is deliberately not."""
    def _sqd(**kwargs):
        from .sqd import SQDSolver
        return SQDSolver(**kwargs)

    def _skqd(**kwargs):
        """SQD driven by time-evolved states — the *same* driver, a different circuit source.

        ⚠ Registered as its own name because that is what the literature calls it and what a
        caller will look for, **not** because it is a second algorithm stack: it constructs
        :class:`~kuiva.qc.sqd.SQDSolver` with
        :class:`~kuiva.qc.ansatz.TimeEvolutionStrategy`, which is exactly the swap
        the design says it should be. An explicit ``strategy=`` overrides it, and then
        the name is a lie, so it is refused.
        """
        from .ansatz import TimeEvolutionStrategy
        from .sqd import SQDSolver

        if "strategy" in kwargs:
            raise ValueError("algorithm 'skqd' *is* the sampled-subspace driver with a "
                             "time-evolution circuit source; passing strategy= would make the "
                             "name mean something else. Use 'sqd' with your own strategy.")
        times = kwargs.pop("times", None)
        steps = kwargs.pop("steps", 1)
        screen = kwargs.pop("screen", 0.0)
        n_elec = kwargs["n_elec"]
        strategy = TimeEvolutionStrategy(
            n_elec=n_elec, steps=steps, screen=screen,
            **({} if times is None else {"times": tuple(float(t) for t in times)}))
        return SQDSolver(strategy=strategy, **kwargs)

    def _vqe(**kwargs):
        from .vqe import VQESolver
        return VQESolver(**kwargs)

    from .sqd import REQUIRED_PRIMITIVES
    from .vqe import REQUIRED_PRIMITIVES as VQE_PRIMITIVES

    register_algorithm("sqd", _sqd, REQUIRED_PRIMITIVES,
                       description="sample-based quantum diagonalization: a circuit as a "
                                   "configuration sampler, the Hamiltonian diagonalized "
                                   "classically in the sampled subspace")
    register_algorithm("skqd", _skqd, REQUIRED_PRIMITIVES,
                       description="sample-based Krylov quantum diagonalization: the same "
                                   "driver sampling Trotterized exp(-iHt)|ref> at a ladder of "
                                   "times, so the circuit comes from the Hamiltonian rather "
                                   "than from an ansatz guess")
    register_algorithm("vqe", _vqe, VQE_PRIMITIVES,
                       description="variational quantum eigensolver: the state stays in the "
                                   "device, its energy is measured and a classical optimizer "
                                   "moves the circuit parameters. Exploratory; its "
                                   "RDM tomography is the cost SQD exists to avoid")


_register_default_algorithms()


__all__ = ["AlgorithmSpec", "algorithm_spec", "available_algorithms", "build_ci_solver",
           "register_algorithm", "resolve_ci_solver", "unregister_algorithm"]
