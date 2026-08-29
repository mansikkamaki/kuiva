"""The instant a run must have stopped by, and the reserve it keeps in order to stop cleanly.

A long calculation ends in one of three ways. It converges; it is killed by a batch
scheduler when the allocation runs out, leaving whatever the last checkpoint held; or it
**stops itself in time**, writes a final checkpoint and exits. This module is the third one.

Three rules shape everything here.

1. ⚠ **There is no default deadline, and there is no default source for one.** A cluster
   with no queue time limit is an ordinary place to run, and a deadline invented for such a
   run could only ever end it early for no reason. ``deadline=None`` — the default
   everywhere — does not look for a limit, does not print a line and never stops anything.
2. ⚠ **The decision is predictive, not reactive.** Stopping when the time is already spent
   is stopping too late: the checkpoint still has to be written, and it is written *after*
   the wall. The rule is therefore *"can one more macro-iteration and its checkpoint still
   finish?"* — :meth:`Deadline.should_stop` compares the time remaining against the longest
   recent iteration plus the estimated write plus a stated safety margin, and stops while
   there is still room to save.
3. ⚠ **An explicitly requested source that cannot be read refuses.** ``deadline="slurm"``
   outside a Slurm job is a mistake worth an exception, not a run that quietly has no
   deadline and is killed twelve hours later. ``deadline="auto"`` is the portable form: it
   uses a queue limit where there is one, states plainly that there is none where there is
   not, and never refuses.

**Granularity, stated honestly.** The deadline acts through the orbital optimizer's
``callback(info)`` seam, so it can stop the run **between macro-iterations and nowhere
else**. One CI solve, one DMRG solve and one NEVPT2 excitation class are uninterruptible;
if a single macro-iteration outlives the allocation, no deadline can help, and the reserve
arithmetic will simply refuse to start another one.

Reading the limit from the queue
--------------------------------
Queue systems are a registry (:data:`QUEUE_SYSTEMS`) of ``(job-id probe, limit probe)``
pairs so a second one is a function and an entry rather than a change here. Slurm is the
one implemented, and it is read in this order:

1. ``$SLURM_JOB_END_TIME`` — the projected end of the allocation as a UNIX timestamp, set
   in the job environment. Free, needs no Slurm client binaries and contacts nothing;
2. ``scontrol show job -o <job id>`` and its ``EndTime=`` field. This asks slurmctld, so it
   is done **once, at construction, never in a loop**: polling the controller from every
   macro-iteration of every job on a cluster is exactly the thing that makes a scheduler
   slow for everyone.

⚠ The limit is read **once**. An allocation extended with ``scontrol update TimeLimit=``
after the run started is not noticed, and the run stops at the limit it was told about —
early rather than late, which is the safe direction of that error.

⚠ ``EndTime=Unknown`` (a job with no time limit) is *unlimited*, and so is any end time
absurdly far in the future: a job whose limit is measured in decades has no limit, and
:data:`UNLIMITED_HORIZON_SECONDS` is where this module stops pretending otherwise. An
unlimited allocation produces a deadline that never fires and says so, which is a different
thing from having asked for none.
"""
from __future__ import annotations

import os
import re
import subprocess
import time
from collections import deque
from typing import Callable, Dict, Optional, Sequence, Tuple

from . import output as out
from .logging import get_logger

log = get_logger(__name__)

#: An end time further away than this is not a limit at all. Slurm reports an unlimited
#: job's end as ``Unknown``, but the environment variable carries a sentinel far in the
#: future instead, and a "deadline" in 2106 would otherwise be carried around as a number.
UNLIMITED_HORIZON_SECONDS = 10.0 * 365.0 * 24.0 * 3600.0

#: Seconds kept back beyond the predicted iteration and the estimated checkpoint write.
#: ⚠ It covers what neither of those measures: the interpreter's own teardown, an HDF5
#: flush slower than the measured sequential bandwidth, and the scheduler's grace between
#: its warning signal and its kill. It is stated in the output rather than hidden.
DEFAULT_SAFETY_SECONDS = 60.0

#: How many recent macro-iterations the prediction looks at. The **longest** of them is
#: used, not the mean: iterations are not uniform (a second-order escalation is several
#: times a quasi-Newton step), and a prediction that is too low is a run that gets killed.
DEFAULT_WINDOW = 3

#: How long to wait for a queue-system client before giving up on it. A scheduler under
#: load can be slow, and a probe that hangs would stall the calculation it was informing.
QUEUE_PROBE_TIMEOUT_SECONDS = 10.0

_DURATION_UNITS = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}
_DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([smhd])", re.IGNORECASE)
#: Slurm's own ``--time`` spellings that are unambiguous: they carry a colon, so they cannot
#: be confused with a plain count of anything. ``[DD-]HH:MM:SS`` and ``MM:SS``.
_CLOCK_RE = re.compile(r"^(?:(\d+)-)?(\d+):(\d{1,2})(?::(\d{1,2}))?$")


def parse_walltime(value) -> float:
    """A wall-time budget in **seconds**, from a number or an explicit string.

    Accepted:

    * a number — seconds;
    * unit-suffixed pieces: ``"6h"``, ``"90m"``, ``"1h30m"``, ``"2d"``, ``"45s"``;
    * the clock forms a batch script already contains: ``"24:00:00"``
      (``HH:MM:SS``), ``"2-12:00:00"`` (``DD-HH:MM:SS``), ``"90:00"`` (``MM:SS``).

    ⚠ **A bare numeric string is refused.** ``"60"`` means sixty *minutes* to Slurm's
    ``--time`` and would mean sixty *seconds* here, and a factor of sixty in a deadline is
    the kind of silent difference that ends an allocation with nothing written. Say
    ``"60m"`` or ``60``.
    """
    if isinstance(value, bool):                       # bool is an int, and never a duration
        raise TypeError("a wall-time budget is a number of seconds or a string such as "
                        "'6h', '90m' or '24:00:00'; got {!r}".format(value))
    if isinstance(value, (int, float)):
        seconds = float(value)
        if seconds <= 0.0:
            raise ValueError("a wall-time budget must be positive; got {!r} s".format(value))
        return seconds
    text = str(value).strip()
    if not text:
        raise ValueError("a wall-time budget cannot be empty")
    clock = _CLOCK_RE.match(text)
    if clock is not None:
        days, first, second, third = clock.groups()
        if third is None:                             # MM:SS
            seconds = float(first) * 60.0 + float(second)
        else:                                         # [DD-]HH:MM:SS
            seconds = float(first) * 3600.0 + float(second) * 60.0 + float(third)
        seconds += float(days or 0) * 86400.0
        if seconds <= 0.0:
            raise ValueError("a wall-time budget must be positive; got {!r}".format(value))
        return seconds
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        raise ValueError(
            "a bare number as a string is ambiguous and is refused: {!r} is sixty minutes to "
            "Slurm's --time and would be sixty seconds here. Write it with a unit ('{}m', "
            "'{}h', '{}s'), as a clock time ('{}:00:00'), or as a number of seconds without "
            "the quotes".format(text, text, text, text, text))
    pieces = _DURATION_RE.findall(text)
    if not pieces or "".join(a + b for a, b in pieces) != re.sub(r"\s+", "", text):
        raise ValueError(
            "could not read {!r} as a wall time: give seconds as a number, a unit-suffixed "
            "string ('6h', '90m', '1h30m'), or a clock time ('24:00:00', '2-12:00:00')"
            .format(value))
    seconds = sum(float(amount) * _DURATION_UNITS[unit.lower()] for amount, unit in pieces)
    if seconds <= 0.0:
        raise ValueError("a wall-time budget must be positive; got {!r}".format(value))
    return seconds


def format_duration(seconds: float) -> str:
    """``"5h 42m"`` — ASCII, coarse, for the output stream (kuiva/util/output.py's rule)."""
    if seconds == float("inf"):
        return "unlimited"
    if seconds < 0.0:
        return "-" + format_duration(-seconds)
    total = int(seconds)
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return "{}d {}h {}m".format(days, hours, minutes)
    if hours:
        return "{}h {}m".format(hours, minutes)
    if minutes:
        return "{}m {}s".format(minutes, secs)
    return "{}s".format(secs)


def format_instant(epoch: float) -> str:
    """A local wall-clock instant, ASCII, no locale-dependent month names."""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(epoch))


# --- reading a limit out of the queue system ---------------------------------------------

class QueueLimit(object):
    """What a queue system says about the end of this allocation.

    ``end_time`` is a UNIX timestamp, or ``None`` when the allocation is ``unlimited``.
    ``source`` names *how* it was read, because "the environment said so" and "the
    controller said so" are different claims with different staleness.
    """

    __slots__ = ("system", "job", "end_time", "unlimited", "source")

    def __init__(self, system: str, job: str, *, end_time: Optional[float] = None,
                 unlimited: bool = False, source: str = "") -> None:
        self.system = system
        self.job = job
        self.end_time = None if unlimited else float(end_time)
        self.unlimited = bool(unlimited)
        self.source = source

    def describe(self) -> str:
        return "{} job {} ({})".format(self.system, self.job, self.source)

    def __repr__(self) -> str:                        # pragma: no cover - debugging aid
        return "QueueLimit({}, job={}, end_time={}, unlimited={})".format(
            self.system, self.job, self.end_time, self.unlimited)


def _run_probe(argv: Sequence[str]) -> Optional[str]:
    """Run a queue-system client and return its stdout, or ``None``.

    ⚠ **Never raises.** A missing binary, a timeout, a controller that says no: all of them
    mean "this source did not answer", and a failure to *read* a limit must not be able to
    fail a calculation. What refuses is one level up, where an explicitly requested source
    that answered nowhere is an error.
    """
    try:
        completed = subprocess.run(list(argv), stdout=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL,
                                   timeout=QUEUE_PROBE_TIMEOUT_SECONDS)
    except Exception as exc:                          # noqa: BLE001 - every failure is a miss
        log.debug("queue probe %s did not answer (%s: %s)", " ".join(argv),
                  type(exc).__name__, exc)
        return None
    if completed.returncode != 0:
        log.debug("queue probe %s exited %d", " ".join(argv), completed.returncode)
        return None
    return completed.stdout.decode("utf-8", "replace")


def _slurm_job_id() -> Optional[str]:
    """This process's Slurm job id, or ``None`` when it is not inside a Slurm job."""
    for name in ("SLURM_JOB_ID", "SLURM_JOBID"):
        value = os.environ.get(name)
        if value:
            return value.strip()
    return None


def _slurm_limit(job: str) -> Optional[QueueLimit]:
    """When this Slurm allocation ends — from the environment first, then ``scontrol``."""
    raw = os.environ.get("SLURM_JOB_END_TIME", "").strip()
    if raw:
        try:
            end = float(raw)
        except ValueError:
            log.debug("SLURM_JOB_END_TIME is %r, which is not a timestamp; falling back to "
                      "scontrol", raw)
        else:
            unlimited = end - time.time() > UNLIMITED_HORIZON_SECONDS
            return QueueLimit("Slurm", job, end_time=end, unlimited=unlimited,
                              source="SLURM_JOB_END_TIME")
    stdout = _run_probe(("scontrol", "show", "job", "-o", str(job)))
    if stdout is None:
        return None
    match = re.search(r"\bEndTime=(\S+)", stdout)
    if match is None:
        log.debug("scontrol output for job %s carries no EndTime field", job)
        return None
    value = match.group(1)
    if value.lower() in ("unknown", "none", "n/a"):
        return QueueLimit("Slurm", job, unlimited=True, source="scontrol EndTime")
    try:
        # ⚠ scontrol prints LOCAL time with no zone, so this is converted through the local
        # zone and is ambiguous by an hour across a DST fold. That is why the environment
        # variable above -- already an unambiguous UNIX timestamp -- is preferred over this.
        end = time.mktime(time.strptime(value, "%Y-%m-%dT%H:%M:%S"))
    except ValueError:
        log.debug("could not read scontrol's EndTime=%r as a local timestamp", value)
        return None
    unlimited = end - time.time() > UNLIMITED_HORIZON_SECONDS
    return QueueLimit("Slurm", job, end_time=end, unlimited=unlimited,
                      source="scontrol EndTime")


#: name -> (job-id probe, limit probe). ⚠ **This is the extension point**: a second queue
#: system (PBS/Torque's ``PBS_JOBID`` + ``qstat``, LSF's ``LSB_JOBID`` + ``bjobs``) is two
#: functions and one entry here, and nothing else in the program learns a scheduler's name.
#: Order matters only for ``"auto"``/``"queue"``, which take the first system that answers.
_JobProbe = Callable[[], Optional[str]]
_LimitProbe = Callable[[str], Optional[QueueLimit]]
QUEUE_SYSTEMS: Dict[str, Tuple[_JobProbe, _LimitProbe]] = {
    "slurm": (_slurm_job_id, _slurm_limit),
}


def detect_queue() -> Optional[str]:
    """The registered queue system whose job this process is running inside, or ``None``."""
    for name, (job_probe, _) in QUEUE_SYSTEMS.items():
        if job_probe() is not None:
            return name
    return None


def queue_limit(system: Optional[str] = None) -> Optional[QueueLimit]:
    """The allocation's end, from ``system`` or from whichever one this job is in.

    Returns ``None`` when there is no such job, or when the system is in no state to say.
    """
    names = list(QUEUE_SYSTEMS) if system is None else [system.lower()]
    for name in names:
        if name not in QUEUE_SYSTEMS:
            raise ValueError("unknown queue system {!r}; this build knows {}"
                             .format(system, ", ".join(sorted(QUEUE_SYSTEMS))))
        job_probe, limit_probe = QUEUE_SYSTEMS[name]
        job = job_probe()
        if job is None:
            continue
        limit = limit_probe(job)
        if limit is not None:
            return limit
        log.debug("inside %s job %s, but its end time could not be read", name, job)
    return None


# --- the deadline itself ------------------------------------------------------------------

class Deadline(object):
    """An instant this run must have stopped by, and the reserve that makes that possible.

    Build one with :meth:`after` (a budget of your own), :meth:`from_queue` (the batch
    allocation's own limit) or :meth:`resolve` (the user-facing ``deadline=`` argument, which
    is what the stage classes call). :meth:`unlimited_from` is the one that never fires.

    The stop rule, evaluated at the end of every macro-iteration::

        remaining  <  longest recent iteration  +  estimated checkpoint write  +  safety

    Every term is measured or estimated rather than assumed: the iteration times come from
    the optimizer's own table through :meth:`observe`, the write estimate from the checkpoint
    policy's measured disk bandwidth, and only :attr:`safety` is a constant — stated in the
    output, and covering what the other two cannot see.
    """

    def __init__(self, *, epoch: Optional[float] = None, duration: Optional[float] = None,
                 unlimited: bool = False, source: str = "explicit",
                 safety: float = DEFAULT_SAFETY_SECONDS,
                 window: int = DEFAULT_WINDOW) -> None:
        if not unlimited and (epoch is None) == (duration is None):
            raise ValueError("a Deadline is an absolute instant (epoch=) or a budget from "
                             "now (duration=), not both and not neither")
        self.unlimited = bool(unlimited)
        self.source = str(source)
        self.safety = float(safety)
        if self.safety < 0.0:
            raise ValueError("the safety margin cannot be negative; got {}".format(safety))
        self._epoch = None if epoch is None else float(epoch)
        self._duration = None if duration is None else float(duration)
        # ⚠ Two clocks, on purpose. An allocation ends at an absolute instant, so it is
        # compared against the wall clock; a budget of "six hours from now" is an elapsed
        # time and is measured on the monotonic clock, which no NTP step or DST change can
        # move underneath it.
        self._t0 = time.monotonic()
        self._started = time.time()
        self._iterations = deque(maxlen=max(1, int(window)))
        self.fired = False

    # -- constructors ---------------------------------------------------------------------

    @classmethod
    def unlimited_from(cls, source: str, **kwargs) -> "Deadline":
        """A deadline that never fires, and says where that verdict came from.

        ⚠ Not the same thing as ``deadline=None``: this one was *asked for* and reports that
        the allocation has no limit, which is a statement a reader of the output wants.
        """
        return cls(unlimited=True, source=source, **kwargs)

    @classmethod
    def after(cls, walltime, **kwargs) -> "Deadline":
        """A budget starting now: ``after("6h")``, ``after(3600)``.

        ⚠ **"Now" is when this object is built**, which for a stage class is when the stage
        is *constructed* rather than when it is run. Build the stage next to its ``run()``,
        or use a queue limit, which is an absolute instant and cannot drift this way.

        See :func:`parse_walltime` for what it accepts and what it refuses.
        """
        seconds = parse_walltime(walltime)
        return cls(duration=seconds, source="requested budget of {}"
                   .format(format_duration(seconds)), **kwargs)

    @classmethod
    def at(cls, epoch: float, **kwargs) -> "Deadline":
        """An absolute UNIX timestamp this run must have stopped by."""
        return cls(epoch=float(epoch), source="requested instant", **kwargs)

    @classmethod
    def from_queue(cls, system: Optional[str] = None, *, required: bool = True,
                   **kwargs) -> Optional["Deadline"]:
        """The batch allocation's own end time.

        ``required`` is the difference between the two user-facing spellings: ``"slurm"`` and
        ``"queue"`` demand a limit and **refuse** when there is none to read, while
        ``"auto"`` reports that there is none and carries on. ⚠ An explicit request that
        silently produced no deadline would be the worst of the three outcomes — the run
        would be killed at the wall with nothing written and nothing in the output ever
        having said the request failed.
        """
        limit = queue_limit(system)
        if limit is None:
            where = "no queue system detected" if system is None else \
                "not inside a {} job, or its end time could not be read".format(system)
            if required:
                raise ValueError(
                    "deadline={!r} asks for this batch allocation's time limit and it could "
                    "not be read ({}). Kuiva reads Slurm's limit from $SLURM_JOB_END_TIME "
                    "and then from 'scontrol show job'; if neither is available here, state "
                    "the budget yourself (deadline='6h'), or use deadline='auto', which uses "
                    "a queue limit where there is one and runs without a deadline where "
                    "there is not".format(system or "queue", where))
            return cls.unlimited_from("no queue limit ({})".format(where), **kwargs)
        if limit.unlimited:
            return cls.unlimited_from("{}: no time limit".format(limit.describe()), **kwargs)
        return cls(epoch=limit.end_time, source=limit.describe(), **kwargs)

    @classmethod
    def resolve(cls, spec, **kwargs) -> Optional["Deadline"]:
        """Turn a user's ``deadline=`` into a :class:`Deadline`, or ``None`` for no deadline.

        ============================  =====================================================
        ``deadline=``                 meaning
        ============================  =====================================================
        ``None`` (the default)        no deadline; nothing is read and nothing is printed
        ``"auto"``                    the queue's limit where there is one, none where not
        ``"queue"``                   as ``"auto"``, but **refuses** when there is no limit
        ``"slurm"``                   that named queue system, and refuses likewise
        ``"6h"`` / ``90`` / ``"24:00:00"``  a budget of your own (:func:`parse_walltime`)
        a :class:`Deadline`           used as it is
        ============================  =====================================================
        """
        if spec is None:
            return None
        if isinstance(spec, Deadline):
            return spec
        if isinstance(spec, str):
            key = spec.strip().lower()
            if key in ("none", "off"):
                return None
            if key == "auto":
                return cls.from_queue(None, required=False, **kwargs)
            if key == "queue":
                return cls.from_queue(None, required=True, **kwargs)
            if key in QUEUE_SYSTEMS:
                return cls.from_queue(key, required=True, **kwargs)
        return cls.after(spec, **kwargs)

    # -- the arithmetic -------------------------------------------------------------------

    def remaining(self) -> float:
        """Seconds left before the deadline; ``inf`` when there is no limit."""
        if self.unlimited:
            return float("inf")
        if self._epoch is not None:
            return self._epoch - time.time()
        return self._duration - (time.monotonic() - self._t0)

    def ends_at(self) -> Optional[float]:
        """The deadline as a UNIX timestamp, or ``None`` when there is no limit."""
        if self.unlimited:
            return None
        if self._epoch is not None:
            return self._epoch
        return time.time() + self.remaining()

    def observe(self, seconds: float) -> None:
        """Record one macro-iteration's wall time — the prediction's only input."""
        try:
            value = float(seconds)
        except (TypeError, ValueError):
            return
        if value > 0.0:
            self._iterations.append(value)

    def predicted(self) -> float:
        """How long the next macro-iteration is expected to take [s].

        The **longest** of the recent ones, not their mean: a second-order escalation or a
        harder CI solve makes one iteration several times another, and under-predicting here
        is the one error that gets the job killed.
        """
        return max(self._iterations) if self._iterations else 0.0

    def reserve(self, write_seconds: float = 0.0) -> float:
        """The time that must still be left for another iteration to be worth starting."""
        return self.predicted() + max(float(write_seconds), 0.0) + self.safety

    def should_stop(self, write_seconds: float = 0.0) -> bool:
        """Is it now clear that one more macro-iteration cannot finish and be saved?"""
        if self.unlimited:
            return False
        return self.remaining() < self.reserve(write_seconds)

    def too_late_to_start(self) -> bool:
        """Is there not even room for the margin — i.e. is starting at all pointless?

        Used before the first iteration, where there is nothing measured to predict from:
        it refuses a run that is certain to be killed rather than spending the allocation
        on work that cannot be written down.
        """
        return not self.unlimited and self.remaining() <= self.safety

    def assert_room(self, what: str = "this stage") -> None:
        """Refuse to start work the deadline makes certain to be killed.

        ⚠ Before the first macro-iteration there is nothing measured to predict from, so the
        only honest test is the margin itself. What it catches is the real case: a job
        resubmitted into the tail of an allocation, where starting means spending what is
        left on work that cannot be written down.
        """
        if self.too_late_to_start():
            raise RuntimeError(
                "{} was not started: {} left before the deadline ({}), which is inside the "
                "{} safety margin, so nothing it computed could be written before the run is "
                "killed. Submit it into a longer allocation, or restart from the checkpoint "
                "of the previous one".format(what, format_duration(self.remaining()),
                                             self.source,
                                             format_duration(self.safety)))

    # -- how it says what it did ----------------------------------------------------------

    def report(self, logger=None) -> None:
        """The one block of output that states the deadline, printed where it is selected.

        ⚠ **An absolute instant is printed only for an absolute deadline.** A budget is
        reported as the budget, not as the clock time it happens to land on: the same script
        run twice would otherwise differ on that line, and an example's committed output
        would carry a timestamp that can never match again.
        """
        logger = logger or log
        rows = [("deadline source", self.source)]
        if self.unlimited:
            rows.append(("time limit", "none: this run will not stop itself"))
        elif self._epoch is not None:
            rows.append(("must have stopped by", format_instant(self._epoch), "",
                         "in {}".format(format_duration(self.remaining()))))
        else:
            rows.append(("wall-clock budget", format_duration(self._duration), "",
                         "from where this deadline was made"))
        if not self.unlimited:
            rows.append(("reserved to stop cleanly",
                         "one macro-iteration + the checkpoint write + {}"
                         .format(format_duration(self.safety))))
        out.entries(logger, rows)

    def announce(self, info: dict, *, wrote: Optional[str] = None,
                 write_seconds: float = 0.0) -> None:
        """The ``WARNING`` that says the deadline stopped the run, and what to do next.

        ⚠ A `WARNING` and not a note: a result that stopped on the clock is not a converged
        one, and every consumer of it has to know that from the output alone.
        """
        self.fired = True
        where = ("the last checkpoint written is {}, and a restart continues from it"
                 .format(wrote) if wrote else
                 "NO checkpoint was written for this run, so nothing of it survives the exit")
        log.warning(
            "stopping at macro-iteration %s to meet the deadline (%s): %s left, and one more "
            "iteration would need about %s (%s for the iteration, %s to write, %s margin). "
            "%s",
            info.get("iteration"), self.source, format_duration(self.remaining()),
            format_duration(self.reserve(write_seconds)),
            format_duration(self.predicted()), format_duration(write_seconds),
            format_duration(self.safety), where)

    def as_callback(self, chain=None):
        """A bare ``callback(info)`` for a run with **no** checkpoint policy to hang on.

        With a checkpoint the deadline belongs to
        :class:`kuiva.io.checkpoint.CheckpointPolicy` instead — the policy is what can
        force the final write *before* the stop, and a deadline whose whole purpose is to
        make that write happen has no business being evaluated after it.
        """
        def callback(info: dict):
            self.observe(info.get("wall", 0.0))
            if not info.get("converged") and self.should_stop():
                self.announce(info)
                return False
            return None if chain is None else chain(info)
        return callback

    def __repr__(self) -> str:                        # pragma: no cover - debugging aid
        if self.unlimited:
            return "Deadline(unlimited, source={!r})".format(self.source)
        return "Deadline(remaining={:.0f} s, source={!r})".format(self.remaining(),
                                                                  self.source)


__all__ = ["DEFAULT_SAFETY_SECONDS", "DEFAULT_WINDOW", "Deadline", "QUEUE_SYSTEMS",
           "QueueLimit", "UNLIMITED_HORIZON_SECONDS", "detect_queue", "format_duration",
           "format_instant", "parse_walltime", "queue_limit"]
