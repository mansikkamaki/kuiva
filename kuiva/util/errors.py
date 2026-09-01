"""Exception types shared across layers that must not import one another.

:class:`SolverFailure` is the contract between *a solver* and *a controller*:
:mod:`kuiva.ci.davidson` and :mod:`kuiva.mcscf.preopt` raise it, and
:mod:`kuiva.mcscf.events` is what gives it meaning. It was originally defined in
``kuiva.mcscf.adaptive`` beside the protocol that documents it, which was right until the
full CI acquired an eigensolver of its own: ``kuiva.mcscf.preopt`` imports ``kuiva.ci``, so
**``kuiva.ci`` may not import ``kuiva.mcscf``**, and an exception type owned by one of two
layers that cannot import each other belongs to neither.

``kuiva.mcscf.adaptive`` re-exports it, so every existing import keeps working and the
protocol documentation stays where the protocol is.

:class:`StateAverageSplit` is the second type, and it is here for the same reason in the
other direction: ``kuiva.rdm`` raises it and the two orbital drivers are what give it
meaning.
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


class StateAverageSplit(SolverFailure, ValueError):
    """The requested state count cuts a degenerate block **of the spectrum at this point**.

    Raised by :func:`kuiva.rdm.rdm.state_average_weights` for the odd-electron refusal of the
    state-averaging gate (the rule and its reasons are in that module). It is deliberately
    *both* exception types, because the refusal is genuinely two statements depending on who
    is asking, and neither reading may be lost:

    * to a **caller** — a CASCI, a direct call, a state count typed by a user — it is a
      ``ValueError`` about the request, exactly as it always was. Nothing that catches or
      asserts on ``ValueError`` sees a change.
    * to a **controller** stepping the orbitals it is a :class:`SolverFailure`, because
      whether a count splits a block depends on the spectrum and the spectrum depends on the
      orbitals: the same count can be a clean boundary at the current point and cut a
      manifold at a trial one. The trial point is then rejected and the trust radius shrinks,
      which is the honest response — the controller has learned that it must not move there,
      not that the calculation is ill-posed. At the starting point there is nothing to reject
      and it propagates, so a count that was wrong from the outset still refuses immediately
      and with the same message.

    ⚠ Being both is what keeps this from being a widening of the drivers' ``except``. They
    catch :class:`SolverFailure` and nothing else, so a shape mismatch, a memory refusal or
    any other bug still propagates rather than being turned into an endless shrinking of the
    trust radius — a run that never converges for a reason nothing prints.
    """


__all__ = ["SolverFailure", "StateAverageSplit"]
