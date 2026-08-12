"""Exception types shared across layers that must not import one another.

Only one type lives here so far, and it is here for a concrete structural reason. :class:`SolverFailure` is the contract between *a solver* and *a controller*:
:mod:`kuiva.ci.davidson` and :mod:`kuiva.mcscf.preopt` raise it, and
:mod:`kuiva.mcscf.events` is what gives it meaning. It was originally defined in
``kuiva.mcscf.adaptive`` beside the protocol that documents it, which was right until the
full CI acquired an eigensolver of its own: ``kuiva.mcscf.preopt`` imports ``kuiva.ci``, so
**``kuiva.ci`` may not import ``kuiva.mcscf``**, and an exception type owned by one of two
layers that cannot import each other belongs to neither.

``kuiva.mcscf.adaptive`` re-exports it, so every existing import keeps working and the
protocol documentation stays where the protocol is.
"""
from __future__ import annotations


class SolverFailure(RuntimeError):
    """A CI or DMRG solve produced no usable answer at this point.

    Raised by an implementation when its own fallbacks are exhausted — an eigensolver that did
    not converge, an iteration cap hit, a space too small for the requested states. The
    controller treats it as a property of the *point*, not of the calculation: a rejected
    trial step, or a refused adoption. At the starting point there is nothing to reject, so it
    propagates.

    ⚠ **Never return an unconverged eigenvector instead.** It is a plausible-looking answer
    that silently poisons the RDMs, the orbital gradient and every macro-iteration after it,
    where a raised failure is information the controller already knows how to use
    (:mod:`kuiva.mcscf.events`).
    """


__all__ = ["SolverFailure"]
