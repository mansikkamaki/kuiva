"""Kernel dispatch shim for the CI hot spots.

A **name -> implementation** registry and a resolver, and nothing else. Callers import the
*name* (``resolve("sigma_gather_f")``), never an implementation, so a compiled backend can be
registered later and every caller and every test picks it up unchanged. That is the backend design's
"narrow, well-documented function interfaces so a compiled backend can replace them per-kernel
without touching callers" expressed in code instead of in prose.

⚠ **This is the interface, not a build system.** No pybind11, no compiler, no dependency — a
C++ backend, if it is ever built, calls :func:`register` at import and nothing else changes.
The shim exists now, before any port, because it is what makes the kernel signatures
*checkable*: :mod:`tests.test_kernel_contracts` walks this registry and asserts the
portability requirements on every entry, and ``tests/`` parametrizes over
:func:`backends_for` so registering a second backend runs the whole suite against it with no
test retrofit.

The portability requirements themselves (plain arrays across the boundary, no hash-based
addressing, flat/rectangular layouts, fixed dtypes, asserted memory layout, caller-provided
output buffers, blocking as a parameter, no logging or timing or resource checks inside the
loop, callbacks only at the outermost level, and a named note wherever a threaded port would
reorder a reduction) live with the kernels they constrain. Each registered implementation
states in its own docstring how it satisfies them.

**Parity tolerance for a future port, fixed in advance** so it cannot be rationalized after
the fact:

* a serial port with the same reduction order must be **bitwise identical** — anything else
  is a defect, not a tolerance question;
* a threaded port that reorders a reduction gets **1e-13 relative**, and only for a kernel
  whose docstring carries an explicit reduction-order note. A kernel with no such note must
  be bitwise.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

#: The backend every kernel has and the one everything falls back to.
DEFAULT_BACKEND = "numpy"


@dataclass(frozen=True)
class KernelSpec:
    """One registered implementation of one kernel."""

    name: str
    backend: str
    impl: Callable

    def __repr__(self) -> str:                                   # pragma: no cover - display
        return "KernelSpec({}/{})".format(self.name, self.backend)


_REGISTRY: Dict[str, Dict[str, KernelSpec]] = {}
_PREFERRED = DEFAULT_BACKEND

#: Whether the compiled-backend gate (kuiva/util/native.py) has run to a decision.
_GATE = {"done": False}


def _ensure_backend_gate() -> None:
    """Trigger the ``KUIVA_KERNELS`` gate once, on first resolution (lazy by design).

    ``import kuiva`` must stay instant and side-effect-free with or without a compiled
    build, so the gate runs here — the choke point every resolution passes through — and
    not at package import. A refused explicit request (``KUIVA_KERNELS=native`` with no
    build) raises here on **every** resolution: the flag is only set once the gate has
    reached a benign decision, so a run can never proceed on a backend nobody asked for.
    """
    if _GATE["done"]:
        return
    from ..util import native

    native.activate()
    _GATE["done"] = True


def register(name: str, backend: str, impl: Callable) -> Callable:
    """Register ``impl`` as the ``backend`` implementation of kernel ``name``.

    Returns ``impl``, so it can be used as a decorator factory at the definition site.
    """
    if not callable(impl):
        raise TypeError("kernel {}/{} is not callable".format(name, backend))
    entries = _REGISTRY.setdefault(name, {})
    if backend in entries and entries[backend].impl is not impl:
        raise ValueError("kernel {}/{} is already registered".format(name, backend))
    entries[backend] = KernelSpec(name=name, backend=backend, impl=impl)
    return impl


def kernel(name: str, backend: str = DEFAULT_BACKEND):
    """Decorator form of :func:`register`."""
    def _decorate(impl: Callable) -> Callable:
        return register(name, backend, impl)
    return _decorate


def kernel_names() -> Tuple[str, ...]:
    """Every registered kernel name, sorted."""
    _ensure_backend_gate()
    return tuple(sorted(_REGISTRY))


def backends_for(name: str) -> Tuple[str, ...]:
    """Backends available for ``name``, sorted, preferred one first.

    Tests parametrize over this so a second backend is exercised automatically.
    """
    _ensure_backend_gate()
    entries = _REGISTRY.get(name)
    if not entries:
        raise KeyError("no kernel registered under {!r}".format(name))
    rest = sorted(b for b in entries if b != _PREFERRED)
    return tuple(([_PREFERRED] if _PREFERRED in entries else []) + rest)


def spec(name: str, backend: str = None) -> KernelSpec:
    """The :class:`KernelSpec` that :func:`resolve` would return."""
    _ensure_backend_gate()
    entries = _REGISTRY.get(name)
    if not entries:
        raise KeyError("no kernel registered under {!r}".format(name))
    if backend is None:
        backend = _PREFERRED if _PREFERRED in entries else backends_for(name)[0]
    if backend not in entries:
        raise KeyError("kernel {!r} has no {!r} backend (have: {})".format(
            name, backend, ", ".join(sorted(entries))))
    return entries[backend]


def resolve(name: str, backend: str = None) -> Callable:
    """The implementation of ``name``, from ``backend`` or from the preferred one.

    ⚠ Callers hold the result only for the duration of one call sequence; they never store it
    on an object that outlives a change of backend, and they never learn which one they got.
    """
    return spec(name, backend).impl


def specs() -> Tuple[KernelSpec, ...]:
    """Every registered implementation, for the contract test to walk."""
    _ensure_backend_gate()
    return tuple(_REGISTRY[name][backend]
                 for name in kernel_names()
                 for backend in sorted(_REGISTRY[name]))


def preferred_backend() -> str:
    return _PREFERRED


def set_preferred_backend(backend: str) -> str:
    """Set the backend :func:`resolve` picks by default; returns the previous one."""
    global _PREFERRED
    previous, _PREFERRED = _PREFERRED, str(backend)
    return previous


__all__ = ["DEFAULT_BACKEND", "KernelSpec", "register", "kernel", "kernel_names",
           "backends_for", "spec", "specs", "resolve", "preferred_backend",
           "set_preferred_backend"]
