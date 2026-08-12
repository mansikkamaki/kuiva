"""The adaptive-solver contract for the shared orbital optimizer.

Why this exists
---------------
:func:`kuiva.mcscf.orbopt.optimize_orbitals` assumes ``E(kappa)`` is **one** smooth surface,
and every piece of machinery in it — the quadratic model, the accept/reject test, the L-BFGS
curvature pairs, the convergence test — is only meaningful under that assumption. A CI that
re-selects its determinant space, or (later) a DMRG that re-distributes its bond dimensions,
does not provide one: the objective is a *family* of smooth surfaces indexed by the solver's
internal space, and the trajectory hops between them. Measured on a Ti(2+) CAS(10,18) proxy,
the hops in the convergence tail are a **single determinant** out of 2000 moving the
state-averaged energy by ``±1e-3 Eh`` — noise/signal ~1 at the point where the optimizer
stalls, three orders of magnitude above where it should converge.

This module defines the interface that lets the optimizer take ownership of those hops
instead of suffering them. It contains **no algorithm**: the controller is
:mod:`kuiva.mcscf.events`, the cheap-CI implementation is
:class:`kuiva.mcscf.preopt.CheapCISolver`, and a DMRG one will be its own class.

The contract
------------
Four methods, replacing the plain ``ci_solver(ints) -> (energy, gamma, gamma2)`` callback of
the smooth-surface optimizer without invalidating it:

``solve(ints)``
    Solve **in the incumbent space** and return ``(energy, gamma, gamma2)`` exactly as the
    plain callback does. This is the only thing called at trial points, and it must be a
    deterministic, smooth function of the integrals — that is what makes a trust-region
    accept/reject test mean anything. ⚠ A solver whose ``solve`` re-selects is the whole
    problem this contract exists to remove; it must not.

``propose(ints)``
    Run a *fresh* selection at these integrals and return a :class:`Proposal`, or ``None`` if
    there is no candidate (a non-adaptive solver, or a selection that reproduced the incumbent
    space). **It must not change the incumbent space** — the controller decides that, after
    comparing the candidate's energy against the incumbent's *at the same integrals*.

``adopt(key)``
    Commit the proposal carrying ``key``. Only ever called with the key of the most recent
    :class:`Proposal`.

``space_key()``
    Identity of the incumbent surface. Two calls returning the same key promise the same
    ``E(kappa)``; a change of key tells the controller its curvature memory belongs to a
    surface that no longer exists (chart-scoped memory).

``solve`` and ``propose`` may raise :class:`SolverFailure`. That is not an error path in the
usual sense: an eigensolver that fails to converge at a trial point is *information about that
point*, and the controller turns it into a rejected step rather than a crashed calculation.

⚠ Wrapping a re-selecting callable in :class:`StaticSolver` does not give it event gating
-----------------------------------------------------------------------------------------
The adapter promises exactly one thing: a plain callable behaves as it does today. If that
callable re-selects internally, ``solve`` is still discontinuous, ``propose`` still returns
``None``, and the controller has nothing to gate — it degrades to the current driver on a
jumpy surface. Getting the benefit means implementing the protocol, not wrapping.

References
----------
* Trust-region globalization and the accept/reject framework the controller sits on top of:
  J. Nocedal, S. J. Wright, "Numerical Optimization", 2nd ed., Springer (2006);
  A. R. Conn, N. I. M. Gould, P. L. Toint, "Trust-Region Methods", SIAM (2000).
* Optimization with inexact or adaptive function information — the framing this contract
  borrows from, though the mechanism here is event control rather than noise modelling:
  R. G. Carter, SIAM J. Numer. Anal. 28, 251 (1991), doi:10.1137/0728014; A. S. Berahas,
  R. H. Byrd, J. Nocedal, SIAM J. Optim. 29, 965 (2019), doi:10.1137/18M1190164.
* The selection whose decision boundaries make the surface piecewise: B. Huron, J. P. Malrieu,
  P. Rancurel, J. Chem. Phys. 58, 5745 (1973), doi:10.1063/1.1679199 (CIPSI); N. M. Tubman
  et al., J. Chem. Phys. 145, 044112 (2016), doi:10.1063/1.4955109 (ASCI).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Hashable, Optional, Tuple

import numpy as np

try:                                                  # pragma: no cover - 3.8+ everywhere here
    from typing import Protocol, runtime_checkable
except ImportError:                                   # pragma: no cover
    from typing_extensions import Protocol, runtime_checkable  # type: ignore

from ..util.errors import SolverFailure
from ..util.logging import get_logger

log = get_logger(__name__)

#: What a plain ``ci_solver`` returns, and what :meth:`AdaptiveCISolver.solve` must return.
RDMResult = Tuple[float, np.ndarray, Optional[np.ndarray]]


#: Re-exported from :mod:`kuiva.util.errors`, where it lives so that :mod:`kuiva.ci` can raise
#: it too: ``kuiva.mcscf.preopt`` imports ``kuiva.ci``, so the dependency may not run the other
#: way. The protocol documentation stays here, with the protocol.
SolverFailure = SolverFailure


@dataclass(eq=False)
class Proposal:
    """A candidate space, evaluated but **not** adopted.

    ``energy``, ``gamma`` and ``gamma2`` are the candidate's values at the *same* integrals
    the incumbent was solved with, which is the only comparison that means anything: the
    surfaces differ, so a candidate energy measured anywhere else is not evidence about this
    point. ``eq=False`` because the arrays make a generated ``__eq__`` ambiguous.
    """

    energy: float
    gamma: np.ndarray
    gamma2: Optional[np.ndarray]
    key: Hashable
    #: Short, printable description for the optimizer's event column and the log.
    label: str = ""


@runtime_checkable
class AdaptiveCISolver(Protocol):
    """The four-method contract described in the module docstring.

    ⚠ ``runtime_checkable`` only checks that the four attributes *exist* — it cannot check
    signatures or, more importantly, that ``solve`` really holds its space fixed. That part
    is a promise the implementation makes.
    """

    def solve(self, ints: Any) -> RDMResult:
        """Energy and state-averaged RDMs in the incumbent space."""

    def propose(self, ints: Any) -> Optional[Proposal]:
        """A freshly selected candidate space at these integrals, or ``None``."""

    def adopt(self, key: Hashable) -> None:
        """Make the proposal carrying ``key`` the incumbent."""

    def space_key(self) -> Hashable:
        """Identity of the incumbent surface."""


class StaticSolver:
    """Adapter presenting a plain ``ci_solver(ints)`` callable as an :class:`AdaptiveCISolver`.

    ``solve`` is the callable, unchanged and unwrapped, so a run through the event-gated
    controller reproduces the current driver's trajectory exactly. ``propose`` returns
    ``None`` always: an event may be *attempted*, but it can never adopt anything, the chart
    key never changes, and the controller collapses to the plain trust-region loop it was
    built around. That is the right behaviour for an exact CI, whose space is fixed by
    construction and for which there is nothing to adapt.

    See the module docstring for the trap: this adapts the *interface*, not the surface.
    """

    #: One key for the whole run — a static solver has exactly one surface.
    KEY = "static"

    def __init__(self, ci_solver: Callable[[Any], RDMResult]):
        if not callable(ci_solver):
            raise TypeError("ci_solver must be callable or implement AdaptiveCISolver; got {}"
                            .format(type(ci_solver).__name__))
        self.ci_solver = ci_solver

    def solve(self, ints: Any) -> RDMResult:
        return self.ci_solver(ints)

    def propose(self, ints: Any) -> Optional[Proposal]:
        return None

    def adopt(self, key: Hashable) -> None:
        raise ValueError("a StaticSolver never proposes a space, so there is nothing to "
                         "adopt (key {!r})".format(key))

    def space_key(self) -> Hashable:
        return self.KEY

    def __repr__(self) -> str:
        return "StaticSolver({!r})".format(getattr(self.ci_solver, "__name__",
                                                   self.ci_solver))


def as_adaptive_solver(ci_solver: Any) -> AdaptiveCISolver:
    """Return ``ci_solver`` if it implements the protocol, else wrap it in a
    :class:`StaticSolver`."""
    if isinstance(ci_solver, AdaptiveCISolver):
        return ci_solver
    return StaticSolver(ci_solver)


def array_key(values: np.ndarray) -> str:
    """A stable, hashable identity for an integer array — the usual ``space_key`` ingredient.

    **Sorted before hashing**, deliberately: a determinant list in a different order spans the
    same space and therefore *is* the same surface, while the CI vector indexed against it is
    not the same object. Chart identity is about the surface.

    A 128-bit BLAKE2b digest rather than :func:`hash` of the bytes, so the key is stable
    across processes (Python salts string and bytes hashing per run) and can be written to a
    checkpoint or a log without becoming meaningless.
    """
    a = np.ascontiguousarray(np.sort(np.asarray(values).ravel()))
    return hashlib.blake2b(a.tobytes(), digest_size=16).hexdigest()


__all__ = ["AdaptiveCISolver", "Proposal", "SolverFailure", "StaticSolver",
           "as_adaptive_solver", "array_key", "RDMResult"]
