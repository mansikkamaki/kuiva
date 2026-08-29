"""Stopping cleanly on a signal: the kill that arrives without having been announced.

:mod:`kuiva.util.deadline` covers the case where the end is *known in advance* — a queue's
time limit, a budget — and stops predictively, while there is still room to write. This
module covers the other one: ``scancel``, a preemption, a node draining, a scheduler's
``SIGTERM`` at the wall. Nothing can be predicted there; what arrives is a request to stop,
and the only question is what the run does with it.

Four rules, and the first is a user decision that predates the code.

1. ⚠ **Handlers are opt-in per run and never a default.** A library that installs signal
   handlers behind your back breaks embedding, test runners and notebooks — a
   ``KeyboardInterrupt`` that no longer interrupts is a genuinely bad surprise. ``signals=``
   is off unless asked for, the handlers are installed for the **duration of one stage** and
   the previous dispositions are restored afterwards, exception or not.
2. ⚠ **The handler sets a flag and does nothing else — no logging, no I/O, no allocation of
   consequence.** A Python signal handler runs between bytecodes in the main thread, so it
   can be interrupting *anything*, including a thread holding the logging lock; logging from
   inside it deadlocks the process it was meant to save. Everything the user sees is printed
   later, at the macro-iteration boundary, from ordinary code.
3. ⚠ **The stop happens at the next macro-iteration boundary, exactly as the deadline's
   does.** One CI solve, one DMRG solve and one NEVPT2 class are uninterruptible; a signal
   that arrives inside one is acted on when it ends. Slurm's ``--signal=B:USR1@<seconds>``
   is how you buy the lead time for that, and the lead has to exceed one macro-iteration
   plus the checkpoint write or the kill lands first anyway.
4. ⚠ **A second signal gets out of the way.** The first is a request; the second is an
   order. On the second delivery the previous dispositions are restored and the signal is
   re-raised, so the process dies exactly as it would have without Kuiva in it. Software
   that swallows a repeated ``SIGTERM`` is software people learn to ``kill -9``.

**The request outlives the stage that caught it**, and that is what makes the run *exit*
rather than merely stop: it is recorded process-wide, so the next long stage refuses to
start instead of beginning hours of work that will be killed in thirty seconds
(:func:`raise_if_pending`). A run that never opted in can never have a pending request, so
this costs a null check and changes nothing for anyone else.
"""
from __future__ import annotations

import os
import signal as _signal
import threading
import time
from typing import Dict, Iterable, Optional, Sequence, Tuple

from .logging import get_logger

log = get_logger(__name__)

#: What ``signals=True`` installs. ``TERM`` is what ``scancel``, ``kill`` and a scheduler at
#: the wall send; ``USR1``/``USR2`` are what Slurm's ``--signal=B:USR1@<seconds>`` sends at a
#: lead time you choose, and nothing else claims them. ⚠ ``INT`` is deliberately **not** here:
#: Ctrl-C is expected to interrupt *now*, and a handler that defers it to the end of a
#: macro-iteration would be the surprise this module exists to avoid. Name it explicitly
#: (``signals=("INT",)``) to accept that trade.
DEFAULT_SIGNALS = ("TERM", "USR1", "USR2")


class StopRequested(RuntimeError):
    """A stop was requested by signal, and the work asked for would not survive it.

    Raised by :func:`raise_if_pending` at the start of a long stage, never in the middle of
    one. A script that wants to end quietly catches it::

        try:
            pt = kuiva.NEVPT2(cas).run()
        except StopRequested:
            sys.exit(0)              # the CASSCF's checkpoint is on disk
    """


class StopRequest(object):
    """The record of a stop request: which signal, when, and how many have arrived."""

    __slots__ = ("signum", "at", "count")

    def __init__(self, signum: int, at: float) -> None:
        self.signum = int(signum)
        self.at = float(at)
        self.count = 1

    @property
    def name(self) -> str:
        try:
            return _signal.Signals(self.signum).name
        except ValueError:                            # pragma: no cover - exotic platform
            return "signal {}".format(self.signum)

    def describe(self) -> str:
        return "{} received at {}".format(
            self.name, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.at)))

    def __repr__(self) -> str:                        # pragma: no cover - debugging aid
        return "StopRequest({}, count={})".format(self.name, self.count)


#: ⚠ Process-level, not per-stage, because a signal is delivered to the *process*. It is what
#: makes a stop propagate past the stage that caught it: the CASSCF stops and writes, and the
#: NEVPT2 after it refuses to start rather than beginning work that will be killed.
_PENDING = None                                       # type: Optional[StopRequest]


def pending() -> Optional[StopRequest]:
    """The stop request this process has received, or ``None``."""
    return _PENDING


def clear() -> None:
    """Forget any pending request.

    For a driver running several calculations in one interpreter — the same reason
    :func:`kuiva.util.resources.clear` exists — and for tests. ⚠ Clearing one that a
    scheduler sent does not make the kill go away.
    """
    global _PENDING
    _PENDING = None


def raise_if_pending(what: str = "this stage") -> None:
    """Refuse to start ``what`` when a stop has already been requested.

    ⚠ **This is the difference between stopping and exiting.** Without it a script whose
    CASSCF stopped on a ``SIGTERM`` walks straight into its NEVPT2 and is killed thirty
    seconds later, mid-write, having spent those seconds on work nobody will see.
    """
    request = _PENDING
    if request is None:
        return
    raise StopRequested(
        "{} was not started: {}, and this process is expected to end. What has already been "
        "computed is in whatever checkpoint the previous stage wrote; restart from it in the "
        "next job. (kuiva.util.signals.clear() forgets the request, which does not make the "
        "kill go away.)".format(what, request.describe()))


def _resolve_signal(spec) -> int:
    """``"TERM"``, ``"SIGTERM"``, ``signal.SIGTERM`` or ``15`` -> the number."""
    if isinstance(spec, int) and not isinstance(spec, bool):
        return int(spec)
    name = str(spec).strip().upper()
    if not name.startswith("SIG"):
        name = "SIG" + name
    try:
        return int(getattr(_signal, name))
    except AttributeError:
        raise ValueError(
            "unknown signal {!r}; give a name such as 'TERM' or 'USR1', or a signal number"
            .format(spec))


class SignalStop(object):
    """Catches a stop request for the duration of one stage, and restores what was there.

    Use it as a context manager (which is what the stage classes do)::

        with SignalStop(("TERM", "USR1")) as stopper:
            ...                                  # stopper.requested is None until one lands

    :attr:`requested` reads the process-level record, so it stays true after the block ends
    — that is deliberate (see the module docstring): the stage stops, and the next one
    refuses to start.
    """

    def __init__(self, signals=True) -> None:
        self.numbers = tuple(resolve_signal_set(signals))
        if not self.numbers:
            raise ValueError("a SignalStop with no signals catches nothing; pass "
                             "signals=True for the default set, or name them")
        self._previous = {}                           # type: Dict[int, object]
        self.installed = False
        self.fired = False
        # ⚠ Checked here, where it is cheap and early, as well as at install time: Python can
        # only install a handler from the main thread, and an explicit request that could not
        # be honoured is refused rather than quietly dropped.
        _assert_main_thread(self.numbers)

    # -- installation ---------------------------------------------------------------------

    def install(self) -> "SignalStop":
        """Install the handlers, remembering what was there before."""
        if self.installed:
            return self
        _assert_main_thread(self.numbers)
        for number in self.numbers:
            self._previous[number] = _signal.getsignal(number)
            _signal.signal(number, self._handle)
        self.installed = True
        log.debug("stop-on-signal armed for %s", ", ".join(name_of(n) for n in self.numbers))
        return self

    def restore(self) -> None:
        """Put the previous dispositions back. Idempotent, and safe to call after a stop."""
        if not self.installed:
            return
        for number, previous in self._previous.items():
            try:
                _signal.signal(number, previous)
            except (ValueError, TypeError, OSError):  # pragma: no cover - exotic platform
                log.debug("could not restore the previous handler for %s", name_of(number))
        self._previous.clear()
        self.installed = False

    def __enter__(self) -> "SignalStop":
        return self.install()

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.restore()
        return False

    # -- the handler ----------------------------------------------------------------------

    def _handle(self, signum, frame) -> None:
        """⚠ **Sets a flag. Nothing else, ever.**

        This runs between bytecodes in the main thread, so it can interrupt any code in the
        process — including code holding the logging lock, where logging from here would
        deadlock the run it was meant to save. Everything a user sees about this signal is
        printed later, from ordinary code, at the macro-iteration boundary.
        """
        global _PENDING
        if _PENDING is not None:
            _PENDING.count += 1
            self._escalate(signum)
            return
        _PENDING = StopRequest(signum, time.time())

    def _escalate(self, signum: int) -> None:
        """The second signal: get out of the way and let it act.

        ⚠ Restores the dispositions that were there *before* Kuiva and re-raises, rather
        than forcing the default — so a process that deliberately ignored ``SIGTERM`` still
        ignores it, and one that would have died still dies, at the instant it was told to.
        Software that swallows a repeated kill is software people learn to ``kill -9``.

        A seam, and the only one in this module: a test cannot let this run.
        """
        self.restore()
        os.kill(os.getpid(), signum)

    # -- what the run asks ----------------------------------------------------------------

    @property
    def requested(self) -> Optional[StopRequest]:
        """The pending stop request, or ``None``. Process-level, and it outlives this stage."""
        return _PENDING

    def should_stop(self, write_seconds: float = 0.0) -> bool:
        """Duck-typed against :class:`kuiva.util.deadline.Deadline` so one call site serves
        both stop causes; a signal knows nothing about how long a write takes."""
        return _PENDING is not None

    def announce(self, info: dict, *, wrote: Optional[str] = None,
                 write_seconds: float = 0.0) -> None:
        """The ``WARNING`` that says a signal stopped the run — printed here, not in the
        handler, for the reason in :meth:`_handle`."""
        self.fired = True
        request = _PENDING
        where = ("the last checkpoint written is {}, and a restart continues from it"
                 .format(wrote) if wrote else
                 "NO checkpoint was written for this run, so nothing of it survives the exit")
        log.warning(
            "stopping at macro-iteration %s: %s. The run stops at this boundary rather than "
            "mid-iteration, which is why it has something to leave behind; a second signal "
            "would not wait. %s",
            info.get("iteration"),
            "a stop was requested" if request is None else request.describe(), where)

    def report(self, logger=None) -> None:
        """The one line of output that states the arrangement, where it is selected."""
        from . import output as out

        out.entry(logger or log, "stop on signal",
                  ", ".join(name_of(n) for n in self.numbers), "",
                  "caught at the next macro-iteration; a second one is not waited for")

    def as_callback(self, chain=None):
        """A bare ``callback(info)`` for a run with no checkpoint policy to hang on."""
        def callback(info: dict):
            if not info.get("converged") and self.requested is not None:
                self.announce(info)
                return False
            return None if chain is None else chain(info)
        return callback

    @classmethod
    def resolve(cls, spec) -> "Optional[SignalStop]":
        """Turn a user's ``signals=`` into a :class:`SignalStop`, or ``None`` for none.

        ``None``/``False`` (the default) installs nothing; ``True`` installs
        :data:`DEFAULT_SIGNALS`; a sequence names them; a :class:`SignalStop` is used as it
        is.
        """
        if spec is None or spec is False:
            return None
        if isinstance(spec, SignalStop):
            return spec
        return cls(spec)

    def __repr__(self) -> str:                        # pragma: no cover - debugging aid
        return "SignalStop({}, installed={}, requested={})".format(
            ", ".join(name_of(n) for n in self.numbers), self.installed, _PENDING)


def stop_context(stopper: "Optional[SignalStop]"):
    """``stopper`` as a context manager, or a do-nothing one when there is none.

    So a driver's ``with`` reads the same whether signals were asked for or not, and the
    handlers cannot outlive the call that installed them however it ends.
    """
    if stopper is not None:
        return stopper
    from contextlib import contextmanager

    @contextmanager
    def _nothing():
        yield None

    return _nothing()


def resolve_signal_set(signals) -> Tuple[int, ...]:
    """``True``, a name, or a sequence of names/numbers -> the signal numbers, deduplicated."""
    if signals is True:
        chosen = DEFAULT_SIGNALS                      # type: Iterable
    elif isinstance(signals, (str, int)) and not isinstance(signals, bool):
        chosen = (signals,)
    elif isinstance(signals, Sequence):
        chosen = signals
    else:
        raise TypeError(
            "signals= takes True (the default set: {}), a signal name, or a sequence of "
            "them; got {!r}".format(", ".join(DEFAULT_SIGNALS), signals))
    numbers = []
    for item in chosen:
        number = _resolve_signal(item)
        if number not in numbers:
            numbers.append(number)
    return tuple(numbers)


def name_of(number: int) -> str:
    try:
        return _signal.Signals(number).name
    except ValueError:                                # pragma: no cover - exotic platform
        return str(number)


def _assert_main_thread(numbers: Sequence[int]) -> None:
    """⚠ Refuse rather than degrade, as an unreadable queue limit does.

    Python can only install a signal handler from the main thread. Warning and carrying on
    would leave the caller believing the run is protected while the very kill they asked to
    survive still ends it mid-iteration.
    """
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError(
            "signals={} was asked for from thread {!r}, and Python can only install a signal "
            "handler on the main thread. Run the calculation on the main thread, or drop "
            "signals= and rely on the checkpoints the run writes anyway"
            .format([name_of(n) for n in numbers], threading.current_thread().name))


__all__ = ["DEFAULT_SIGNALS", "SignalStop", "StopRequest", "StopRequested", "clear",
           "name_of", "pending", "raise_if_pending", "resolve_signal_set", "stop_context"]
