"""Heartbeat reporting and progress checking for long-running runs.

Why this exists
---------------
A run that is computing, a run that is spinning in a loop, and a run that never started all
look the same from outside: a process that exists (or does not) and an output file that has
not changed. Identifying runs by their **command line** — ``pgrep -f script.py`` — makes it
worse, because the wrapper shell that launched the script also contains the script's name, so
the pattern matches watchers and launchers as readily as the job. Three incidents in one
session came from exactly that, one of them costing 86 minutes waiting on a script that had
never launched.

The fix is that **the running process reports on itself**. It appends one line per
macro-iteration to a small text file; a checker reads the last line and compares it with what
it saw before. That distinguishes the cases that matter:

* nothing written after the startup allowance -> **never started**
* PID gone from ``/proc``                     -> **died**
* PID alive, counter unchanged, CPU rising    -> **grinding in place**
* PID alive, counter unchanged, CPU flat      -> **blocked / deadlocked**
* counter advanced                            -> **advancing**, with a rate and an ETA

The last two are the ones no liveness check can tell apart, and they are the ones that waste
the most time.

Usage
-----
In the run::

    from progress import Heartbeat
    hb = Heartbeat("bench_orbopt", budget_seconds=600)
    for it in range(max_iter):
        ...
        hb.tick(it, grad=gnorm, energy=e)          # writes a line, returns False if over budget
        if hb.expired:
            break                                  # the budget is the design, not a hope

From a checker (a separate, short-lived command)::

    python -m progress check bench_orbopt

which prints one line of verdict and exits non-zero if the run is not advancing.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

#: Where heartbeats live. Kept out of the source tree; one file per named run.
HEARTBEAT_DIR = Path(os.environ.get("KUIVA_RUN_DIR", "/tmp/kuiva-runs"))
#: Grace period before "no lines yet" counts as "never started" [s].
STARTUP_ALLOWANCE = 120.0
#: A heartbeat older than this, with no live process, is reported as **stale** rather than as
#: a result. Names get reused between a probe and the real run, and a leftover "done" from an
#: earlier probe is indistinguishable from the current run having finished — which is exactly
#: the class of mistake the heartbeat discipline exists to prevent.
STALE_AFTER = 900.0
#: Default wall budget for an ad-hoc run [s] — the ten-minute ad-hoc rule.
DEFAULT_BUDGET = 600.0
_CLOCK_TICKS = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100


def _path(name: str) -> Path:
    return HEARTBEAT_DIR / (name + ".jsonl")


class Heartbeat:
    """Appends one JSON line per tick, and enforces a wall budget.

    The budget is enforced *inside the run* rather than by an external timeout, so a run that
    hits it exits cleanly with everything it has computed so far already on disk. An external
    ``timeout`` leaves nothing behind, which is how 40 minutes of results were lost once.
    """

    def __init__(self, name: str, budget_seconds: float = DEFAULT_BUDGET,
                 meta: Optional[dict] = None):
        self.name = name
        self.budget = float(budget_seconds)
        self.path = _path(name)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        self.n_ticks = 0
        # Truncate: a stale file from a previous run is worse than no file, because it looks
        # like progress that is not happening.
        with self.path.open("w") as fh:
            fh.write(json.dumps({"event": "start", "pid": os.getpid(), "t": self.t0,
                                 "budget": self.budget, "meta": meta or {}}) + "\n")

    @property
    def elapsed(self) -> float:
        return time.time() - self.t0

    @property
    def expired(self) -> bool:
        return self.elapsed > self.budget

    def tick(self, iteration: int, **fields) -> bool:
        """Record one unit of progress. Returns ``False`` once the budget is spent."""
        self.n_ticks += 1
        rec = {"event": "tick", "pid": os.getpid(), "t": time.time(),
               "elapsed": self.elapsed, "iteration": int(iteration)}
        rec.update({k: (float(v) if isinstance(v, (int, float)) else v)
                    for k, v in fields.items()})
        with self.path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        return not self.expired

    def finish(self, **fields) -> None:
        rec = {"event": "done", "pid": os.getpid(), "t": time.time(),
               "elapsed": self.elapsed}
        rec.update(fields)
        with self.path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")


# --- checking ------------------------------------------------------------------------------

def _proc_utime(pid: int) -> Optional[float]:
    """CPU seconds burned by ``pid``, or ``None`` if it is gone. Reads ``/proc`` directly, so
    it identifies the process by **PID**, never by a command-line pattern."""
    try:
        stat = Path("/proc/%d/stat" % pid).read_text()
    except (OSError, ValueError):
        return None
    close = stat.rfind(")")
    fields = stat[close + 2:].split()
    try:
        return (int(fields[11]) + int(fields[12])) / _CLOCK_TICKS
    except (IndexError, ValueError):
        return None


@dataclass
class Status:
    verdict: str                 # never-started | died | grinding | blocked | advancing | done
    detail: str
    iteration: int = -1
    elapsed: float = 0.0
    advancing: bool = False

    def __str__(self) -> str:
        return "[%s] %s" % (self.verdict, self.detail)


def check(name: str, previous: Optional[dict] = None) -> Status:
    """Classify a run. ``previous`` is the ``as_dict()`` of an earlier check, if any."""
    path = _path(name)
    if not path.exists():
        return Status("never-started", "no heartbeat file at %s" % path)
    lines = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    if not lines:
        return Status("never-started", "heartbeat file is empty")
    start = lines[0]
    pid = int(lines[-1].get("pid", start.get("pid", -1)))
    ticks = [r for r in lines if r.get("event") == "tick"]
    done = [r for r in lines if r.get("event") == "done"]
    age = time.time() - float(start.get("t", time.time()))

    alive = _proc_utime(pid) is not None
    if done:
        stale = "" if (alive or age < STALE_AFTER) else \
            "  ** STALE: started %.0f min ago, pid %d gone — this is a leftover file, not " \
            "the run you are watching **" % (age / 60.0, pid)
        return Status("stale" if stale else "done",
                      "finished after %.0f s, %d ticks (started %.0f s ago)%s"
                      % (done[-1]["elapsed"], len(ticks), age, stale),
                      advancing=False, iteration=ticks[-1]["iteration"] if ticks else -1)
    if not ticks:
        if not alive:
            return Status("died", "pid %d gone before the first tick" % pid)
        if age > STARTUP_ALLOWANCE:
            return Status("blocked", "pid %d alive but no tick in %.0f s (startup allowance "
                                     "%.0f s)" % (pid, age, STARTUP_ALLOWANCE))
        return Status("advancing", "starting up (%.0f s)" % age, advancing=True)

    last = ticks[-1]
    if not alive:
        return Status("died", "pid %d gone after %d ticks (last iteration %d)"
                      % (pid, len(ticks), last["iteration"]),
                      iteration=last["iteration"], elapsed=last["elapsed"])

    if previous is None:
        return Status("advancing", "%d ticks, at iteration %d after %.0f s"
                      % (len(ticks), last["iteration"], last["elapsed"]),
                      iteration=last["iteration"], elapsed=last["elapsed"], advancing=True)

    moved = last["iteration"] != previous.get("iteration")
    if moved:
        d_it = last["iteration"] - previous.get("iteration", 0)
        d_t = last["elapsed"] - previous.get("elapsed", 0.0)
        rate = d_it / d_t if d_t > 0 else float("nan")
        return Status("advancing", "iteration %d (+%d in %.0f s, %.2f it/s)"
                      % (last["iteration"], d_it, d_t, rate),
                      iteration=last["iteration"], elapsed=last["elapsed"], advancing=True)
    # Not moving: is it burning CPU or stuck?
    cpu_now = _proc_utime(pid)
    cpu_before = previous.get("utime")
    if cpu_before is not None and cpu_now is not None and cpu_now - cpu_before > 1.0:
        return Status("grinding", "pid %d stuck at iteration %d but burning CPU (+%.0f s) — "
                                  "one iteration is taking too long, or it is looping"
                      % (pid, last["iteration"], cpu_now - cpu_before),
                      iteration=last["iteration"], elapsed=last["elapsed"])
    return Status("blocked", "pid %d stuck at iteration %d with no CPU progress"
                  % (pid, last["iteration"]), iteration=last["iteration"],
                  elapsed=last["elapsed"])


def snapshot(name: str) -> dict:
    """State to pass as ``previous`` to the next :func:`check`."""
    path = _path(name)
    if not path.exists():
        return {}
    lines = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    ticks = [r for r in lines if r.get("event") == "tick"]
    pid = int(lines[-1].get("pid", -1)) if lines else -1
    out = {"utime": _proc_utime(pid)}
    if ticks:
        out.update({"iteration": ticks[-1]["iteration"], "elapsed": ticks[-1]["elapsed"]})
    return out


def main(argv) -> int:
    if len(argv) < 3 or argv[1] != "check":
        print("usage: python -m progress check <name> [--wait SECONDS]", file=sys.stderr)
        return 2
    name = argv[2]
    wait = 0.0
    if "--wait" in argv:
        wait = float(argv[argv.index("--wait") + 1])
    before = snapshot(name)
    if wait > 0:
        time.sleep(wait)
    st = check(name, before if wait > 0 else None)
    print(st)
    return 0 if st.verdict in ("advancing", "done") else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
