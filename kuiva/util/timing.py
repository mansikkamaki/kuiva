"""Context-manager timers and the run-time summary.

What is measured, and why both
------------------------------
Every timed region records **wall time and CPU time**. This is not redundancy:

* ``wall`` is what the user waits for;
* ``cpu`` (``time.process_time``: user+system CPU summed over all threads of the process) is
  what the calculation actually costs.

On the development machine these differ by a large factor: ``thermald``
clamps the package under sustained load by injecting idle cycles, so a kernel can take three
times longer in wall time than its arithmetic warrants. A profile ranked by wall time on that
machine is ranked partly by the thermal envelope. **Decide which kernels to port to C++
on CPU seconds, never on wall seconds** — this is why the summary table prints both, plus
their ratio.

Reading the ratio ``cpu/wall``:

* ``~= n_threads``  — the region was compute-bound and parallel; the wall time is honest.
* ``~= 1``          — serial region (a Python loop, or a single-threaded LAPACK call).
* ``<  1``          — the process was waiting: I/O, an external program, or the CPU being
  clamped. A long region with a ratio below one on a threaded kernel is the signature of
  thermal throttling, and the summary footer says so.

.. warning::
   ``process_time`` counts CPU across **all threads of the process**, including MKL/OpenMP
   workers that are *spin-waiting* rather than working. With the default ``KMP_BLOCKTIME``
   (200 ms) the threads keep spinning after a parallel region ends, so a short region that
   merely follows a threaded BLAS call is charged for their idling: measured here at 0.35 s
   of "CPU" for a region that did nothing but sleep for 0.05 s. Consequences:
   **(a)** short regions (below roughly a second) sandwiched between threaded ones have
   inflated CPU times and their ``cpu/wall`` may exceed the true thread count; **(b)** when
   profiling to decide on a C++ port, compare *long* regions, or set ``KMP_BLOCKTIME=0``
   for the profiling run. This is a property of the threading runtime, not of the timer, and
   it cannot be fixed here — per-thread accounting would have to come from the kernel that
   ran the work.

Usage
-----
::

    from ..util.timing import timer, timed, summary

    with timer("integrals/cholesky"):
        ...                                  # nests automatically inside an enclosing timer

    @timed("orth/canonical")
    def canonical_orthogonalization(...):
        ...

    summary(log)                             # one table at the end of the run

Timers nest by call order, and the registry keys on the *path* (``casscf/ci/sigma``), so the
same kernel called from two places is reported separately — which is usually what you want to
know. Times are inclusive of nested regions; the summary indents children under their parent
so the difference is readable.

The registry is process-global and **not thread-safe**: Kuiva's parallelism is threaded BLAS
underneath serial Python orchestration, so timers are only ever entered from the main
thread. If that ever changes, make ``_STACK`` thread-local.
"""
from __future__ import annotations

import functools
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import output as out
from .logging import get_logger

log = get_logger(__name__)


@dataclass
class TimingNode:
    """Accumulated wall/CPU time for one timed region, identified by its path."""
    path: str
    label: str
    depth: int
    calls: int = 0
    wall: float = 0.0
    cpu: float = 0.0
    order: int = 0

    @property
    def parallel_ratio(self) -> float:
        """``cpu/wall`` — effective threads used; see the module docstring."""
        return self.cpu / self.wall if self.wall > 0.0 else 0.0


class TimingRegistry:
    """Ordered collection of :class:`TimingNode`, with the active nesting stack."""

    def __init__(self) -> None:
        self._nodes: Dict[str, TimingNode] = {}
        self._stack: List[str] = []
        self._counter = 0

    # -- stack management (used by Timer) --
    def push(self, label: str) -> str:
        path = "/".join(self._stack + [label])
        self._stack.append(label)
        if path not in self._nodes:
            self._counter += 1
            self._nodes[path] = TimingNode(path=path, label=label,
                                           depth=len(self._stack) - 1, order=self._counter)
        return path

    def pop(self, path: str, wall: float, cpu: float) -> TimingNode:
        if self._stack:
            self._stack.pop()
        node = self._nodes[path]
        node.calls += 1
        node.wall += wall
        node.cpu += cpu
        return node

    # -- inspection --
    def nodes(self) -> List[TimingNode]:
        """Nodes in call order (parents before their children)."""
        return sorted(self._nodes.values(), key=lambda n: n.order)

    def get(self, path: str) -> Optional[TimingNode]:
        return self._nodes.get(path)

    def total_wall(self) -> float:
        """Wall time of the top-level regions only (children are inclusive in them)."""
        return sum(n.wall for n in self._nodes.values() if n.depth == 0)

    def total_cpu(self) -> float:
        return sum(n.cpu for n in self._nodes.values() if n.depth == 0)

    def clear(self) -> None:
        self._nodes.clear()
        self._stack.clear()
        self._counter = 0


#: The process-global registry. Modules time against this unless handed another one.
REGISTRY = TimingRegistry()


class Timer:
    """Context manager timing a region in wall and CPU seconds.

    ``with timer("label") as t:`` ... ``t.wall``/``t.cpu`` are available afterwards, so a
    caller can put the cost of a step straight into its own output table without asking the
    registry. Exceptions do not lose the measurement (the time is recorded in ``__exit__``).
    """

    __slots__ = ("label", "registry", "log", "level", "_path", "_w0", "_c0", "wall", "cpu")

    def __init__(self, label: str, *, registry: Optional[TimingRegistry] = REGISTRY,
                 log: Optional[logging.Logger] = None, level: int = logging.DEBUG):
        self.label = label
        self.registry = registry
        self.log = log
        self.level = level
        self._path = label
        self.wall = 0.0
        self.cpu = 0.0

    def __enter__(self) -> "Timer":
        if self.registry is not None:
            self._path = self.registry.push(self.label)
        self._w0 = time.perf_counter()
        self._c0 = time.process_time()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.wall = time.perf_counter() - self._w0
        self.cpu = time.process_time() - self._c0
        if self.registry is not None:
            self.registry.pop(self._path, self.wall, self.cpu)
        target = self.log if self.log is not None else log
        # timings log at DEBUG and are summarised at INFO on completion.
        target.log(self.level, "%s: %.3f s wall, %.3f s cpu (x%.2f)",
                   self._path, self.wall, self.cpu,
                   self.cpu / self.wall if self.wall > 0 else 0.0)
        return False


def timer(label: str, **kwargs) -> Timer:
    """Shorthand for ``Timer(label)``; see :class:`Timer`."""
    return Timer(label, **kwargs)


def timed(label: Optional[str] = None, **kwargs):
    """Decorator timing a whole function under ``label`` (default: its qualified name)."""

    def decorate(func):
        name = label or func.__qualname__

        @functools.wraps(func)
        def wrapper(*args, **kw):
            with Timer(name, **kwargs):
                return func(*args, **kw)

        return wrapper

    return decorate


def summary(logger: Optional[logging.Logger] = None, *,
            registry: TimingRegistry = REGISTRY, min_seconds: float = 0.0,
            title: str = "Timings") -> None:
    """Log the accumulated timing table at INFO (summary printed on completion).

    ``min_seconds`` suppresses regions below a wall-time floor, so a long run's summary stays
    a summary. Nested regions are indented under their parent and their times are inclusive.
    """
    logger = logger or log
    nodes = [n for n in registry.nodes() if n.wall >= min_seconds]
    if not nodes:
        return
    total_wall = registry.total_wall()
    out.section(logger, title)
    name_w = max(30, min(40, max(len(n.label) + 2 * n.depth for n in nodes)))
    table = out.Table(logger, [
        out.Column("region", "{}", name_w, align="<"),
        out.col_count("calls", 6),
        out.Column("wall [s]", out.TIME_FMT, 11),
        out.Column("cpu [s]", out.TIME_FMT, 12),
        out.Column("cpu/wall", "{:.2f}", 9),
        out.Column("% wall", "{:.1f}", 7),
    ])
    table.start()
    for n in nodes:
        pct = 100.0 * n.wall / total_wall if total_wall > 0 else 0.0
        table.row("  " * n.depth + n.label, n.calls, n.wall, n.cpu, n.parallel_ratio, pct)
    table.end()
    out.entry(logger, "total (top-level regions)", registry.total_wall(), "s wall",
              fmt=out.TIME_FMT)
    out.entry(logger, "total CPU", registry.total_cpu(), "s cpu", fmt=out.TIME_FMT)
    # Make the interpretation explicit rather than leaving it to be rediscovered.
    out.note(logger, "cost is cpu [s]; wall [s] also reflects thread count and, on a")
    out.note(logger, "thermally clamped machine, time the CPU was forced to stay idle.")


def reset(registry: TimingRegistry = REGISTRY) -> None:
    """Clear accumulated timings (tests, and successive runs in one process)."""
    registry.clear()


__all__ = ["Timer", "TimingNode", "TimingRegistry", "REGISTRY",
           "timer", "timed", "summary", "reset"]
