"""Thread control: one knob, applied per region, and proved by measurement.

Two facts about this project make a single global thread count wrong, and both are
measured rather than assumed:

* dense-linear-algebra stages (integral transform, orbital optimizer, sigma builds) want
  **threaded BLAS** and nothing else;
* the tensor-network layer wants the opposite — threaded BLAS buys it nothing (identical
  wall time at 1 and at 8 threads, ``cpu/wall`` of 4 that is pure spin-wait;
  measured) while its *own* compiled kernels parallelize
  well over the pair table.

So the knob is one number — the budget — and the *region* decides who spends it.

The knob
--------
``KUIVA_NUM_THREADS``: one integer, the total the calculation may use. Precedence::

    explicit argument > KUIVA_NUM_THREADS > OMP_NUM_THREADS > all cores

read **once per process** and cached (a kernel caller may sit inside a sweep loop, and
portability rule B8 forbids anything slower than a lookup there). "All cores" means the
CPUs this process is actually allowed to run on (``sched_getaffinity``), so a cgroup or a
``taskset`` is respected rather than overridden.

⚠ **The budget is applied to MKL only when it came from KUIVA_NUM_THREADS** (or an
explicit :func:`set_budget`). A budget inherited from ``OMP_NUM_THREADS`` or from the core
count leaves the BLAS exactly as the environment configured it — silently overriding a
deliberate ``MKL_NUM_THREADS`` would be the same class of surprise this module exists to
remove.

The regions
-----------
:func:`blas_region` — MKL gets the budget, kernels get 1.
:func:`kernel_region` — kernels get the budget, MKL is clamped to 1 for the duration
(``MKL_Set_Num_Threads_Local``, restored on exit; it is a *local* setting, so a region on
one thread never disturbs another).

Both are entered **once per solve/sweep, never inside a loop** (B8), and they nest: a
DMRG solver entering its kernel region inside the orbital optimizer's BLAS region gets
kernel threads for the sweep and hands the BLAS back on the way out. Compiled kernels
still take their thread count as an **explicit argument** (B7 applied to threads — never
``omp_set_num_threads``, never an environment read inside a kernel); what the region
changes is what :func:`thread_count` hands those callers.

⚠ **Outside every region, ``thread_count()`` is the full budget** — that is the behaviour
that predates the regions, and it is deliberately left alone, because since the
compiled backend landed the un-regioned path is a *threaded-kernel* path (a DMRG
contraction, a determinant scan). Defaulting to BLAS-region behaviour would have
silently serialized them.

The probe
---------
:func:`threads_report` identifies the BLAS actually loaded into the process (from
``/proc/self/maps``, not from a build flag), measures one small ``zgemm`` at 1 thread and
at the budget, and **warns** in the two directions that otherwise fail silently:

* budget > 1 but no measured speedup — CPU-hours spent on threads that are not there;
* ``cpu/wall`` far above 1 at nominal one thread — a second threading runtime spinning
  next to MKL's (the libgomp trap of the build script) or a profiling run with
  ``KMP_BLOCKTIME`` unset, which corrupts every ``cpu_seconds`` figure taken afterwards.

It runs once per process from the run banner, and standalone as::

    python -m kuiva.util.threads

⚠ On the development box the probe measures a thermally throttled machine. That
is why it reports a **ratio** — both sides are equally hot — and never an absolute
GFLOP/s. The measurement is sized to a few milliseconds, small enough that no run has to
budget for it and large enough to resolve a 4x speedup by an order of magnitude.

⚠ **Its size is FIXED, and every allocation it makes is the same on every run** — see
:data:`_PROBE_DIM`. A diagnostic that runs before the calculation is not entitled to make
the calculation irreproducible, and an adaptive one did exactly that (the measurement is in
measured).

⚠ **Not a dependency, deliberately**: ``threadpoolctl`` does exactly this and does it well,
but the project pins one BLAS and needs three calls against it, which is not worth a new
pinned dependency on the default path. Revisit only if a second BLAS ever becomes supported.

References
------------------
Intel oneAPI Math Kernel Library, "Techniques to Set the Number of Threads" and
"MKL_Set_Num_Threads_Local"; the OpenMP composability/nesting notes that the same
documentation set carries. The ctypes binding below uses the **C** entry points
(``MKL_Set_Num_Threads_Local``): ⚠ the lowercase ``mkl_set_num_threads_local`` is the
*Fortran* entry point and takes its argument **by pointer** — calling it by value
segfaults, which is a five-minute mistake to make and a long one to diagnose.
"""
from __future__ import annotations

import ctypes
import os
import time
from typing import NamedTuple, Optional

from .logging import get_logger

log = get_logger(__name__)

#: Below this measured wall speedup, a budget > 1 is declared not to be buying threads.
#: Not a performance target: 1.25 is far below any real threading gain (a 4-thread zgemm
#: measures 3.5-4.4x here) and far above the noise of a millisecond-scale measurement, so
#: the warning fires on "not threaded at all" and on nothing else.
THREADED_SPEEDUP_MIN = 1.25

#: Above this ``cpu/wall`` at nominal one thread, a second runtime is spinning.
SERIAL_CPU_PER_WALL_MAX = 1.5

#: The probe GEMM's dimension and repeat count. ⚠ **Both are FIXED, and that is a
#: correctness requirement, not a simplification.** An adaptive size (grow until the serial
#: call takes some target time) makes the number and size of the probe's allocations depend
#: on how fast the machine happened to be on that run — which moves every later array in
#: the process and, through alignment-dependent BLAS kernels, perturbs the last bits of
#: every subsequent result *differently on every run*. A calculation that is not
#: reproducible on its own machine is a high price for a diagnostic. 320 costs ~7 ms
#: serial here and resolves a 4-thread speedup as 3.5x; best-of-3 removes the upward tail
#: of a millisecond-scale timing without adding a data-dependent branch.
_PROBE_DIM = 320
_PROBE_REPEATS = 3

_BUDGET = None            # type: Optional[int]
_SOURCE = None            # type: Optional[str]
_APPLIED = False          # the budget has been pushed to MKL (only if it is ours)
_REPORT = None            # type: Optional["ThreadsReport"]

# The active region's kernel thread count; None means "no region" (see the module
# docstring: that is the full budget, not 1). A plain module global rather than a
# thread-local by decision — a region is entered once per solve on the driver thread, and
# a kernel called from a worker thread must see the same policy, not a fresh default.
_REGION_KERNEL_THREADS = None     # type: Optional[int]


# --- the budget -------------------------------------------------------------------------


def _all_cores() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))       # respects taskset / cgroups
    except AttributeError:                               # pragma: no cover - non-Linux
        return max(1, os.cpu_count() or 1)


def _positive(value, origin: str) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise ValueError("{}={!r} is not an integer".format(origin, value))
    if n < 1:
        raise ValueError("{}={} must be a positive integer".format(origin, n))
    return n


def _resolve() -> None:
    global _BUDGET, _SOURCE
    for variable in ("KUIVA_NUM_THREADS", "OMP_NUM_THREADS"):
        raw = os.environ.get(variable)
        if raw is None:
            continue
        _BUDGET = _positive(raw, variable)
        _SOURCE = variable
        return
    _BUDGET = _all_cores()
    _SOURCE = "all cores"


def budget() -> int:
    """The total thread budget of this calculation (the resolved knob)."""
    if _BUDGET is None:
        _resolve()
    return int(_BUDGET)


def budget_source() -> str:
    """Where the budget came from: an env var name, ``"all cores"`` or ``"explicit"``."""
    if _BUDGET is None:
        _resolve()
    return str(_SOURCE)


def set_budget(n_threads: int) -> int:
    """Set the budget from the API, overriding the environment. Returns the new value.

    The top of the precedence chain, for a caller that knows better than the environment
    (a benchmark sweeping thread counts, an embedding application). Applies to the BLAS as
    an explicit ``KUIVA_NUM_THREADS`` would.
    """
    global _BUDGET, _SOURCE, _APPLIED, _REPORT
    _BUDGET = _positive(n_threads, "n_threads")
    _SOURCE = "explicit"
    _APPLIED = False
    _REPORT = None
    apply_budget()
    return _BUDGET


def thread_count(explicit: Optional[int] = None) -> int:
    """The thread budget a kernel caller should pass as its explicit thread argument.

    ``explicit`` wins; inside a region the region's policy applies; outside every region
    it is the full budget (module docstring).
    """
    if explicit is not None:
        return _positive(explicit, "thread count")
    if _REGION_KERNEL_THREADS is not None:
        return int(_REGION_KERNEL_THREADS)
    return budget()


# --- MKL, through ctypes on the already-loaded library ------------------------------------


class _MKL(object):
    """The four calls this module needs, bound to the MKL already in the process.

    ``dlopen`` of ``libmkl_rt`` returns the handle of the library NumPy has already
    loaded, so the settings below act on the one MKL of the process — which is the whole
    point of the build script's "one MKL, no libmkl_sequential" rule.
    """

    def __init__(self, lib):
        self._lib = lib
        self._set_local = lib.MKL_Set_Num_Threads_Local
        self._set_local.restype = ctypes.c_int
        self._set_local.argtypes = [ctypes.c_int]
        self._set_global = lib.MKL_Set_Num_Threads
        self._set_global.restype = None
        self._set_global.argtypes = [ctypes.c_int]
        self._max = lib.MKL_Get_Max_Threads
        self._max.restype = ctypes.c_int
        self._max.argtypes = []
        self._version = lib.MKL_Get_Version_String
        self._version.restype = None
        self._version.argtypes = [ctypes.c_char_p, ctypes.c_int]

    def set_local(self, n_threads: int) -> int:
        """Set this thread's MKL width; returns the previous value (0 = follow global)."""
        return int(self._set_local(int(n_threads)))

    def set_global(self, n_threads: int) -> None:
        self._set_global(int(n_threads))

    def max_threads(self) -> int:
        return int(self._max())

    def version(self) -> str:
        buf = ctypes.create_string_buffer(256)
        self._version(buf, 256)
        return buf.value.decode("ascii", "replace").strip()


_MKL_HANDLE = None            # type: Optional[_MKL]
_MKL_LOOKED = False


def _mkl() -> Optional[_MKL]:
    """The process's MKL, or ``None`` if the loaded BLAS is not MKL (never raises)."""
    global _MKL_HANDLE, _MKL_LOOKED
    if _MKL_LOOKED:
        return _MKL_HANDLE
    _MKL_LOOKED = True
    _ensure_blas_loaded()
    if "mkl" not in _loaded_libraries():
        return None
    for name in ("libmkl_rt.so.2", "libmkl_rt.so.1", "libmkl_rt.so"):
        try:
            _MKL_HANDLE = _MKL(ctypes.CDLL(name))
            return _MKL_HANDLE
        except (OSError, AttributeError):
            continue
    log.debug("MKL is mapped into the process but libmkl_rt could not be bound; "
              "thread regions degrade to no-ops")
    return None


def _ensure_blas_loaded() -> None:
    """Make the BLAS be in the process before asking which BLAS is in the process.

    ⚠ The answer is otherwise **order-dependent**: nothing here imports NumPy, so a probe
    running before the first array operation reads a map with no BLAS in it, caches
    "unknown" and never looks again. NumPy links its BLAS at import and dispatches the
    kernel on first use, so one 2x2 complex product settles both.
    """
    try:
        import numpy as np

        np.zeros((2, 2), dtype=np.complex128).dot(np.zeros((2, 2), dtype=np.complex128))
    except Exception:                                     # noqa: BLE001 - identification only
        pass


def _loaded_libraries() -> str:
    """The process's mapped shared objects, lower-cased, as one searchable string.

    Measured, not assumed: what matters is the BLAS **in this process**, which a build
    flag, a ``numpy.__config__`` entry or an environment variable can all misreport.
    """
    try:
        with open("/proc/self/maps", "r") as handle:
            return handle.read().lower()
    except OSError:                                       # pragma: no cover - non-Linux
        return ""


def blas_identity() -> str:
    """A short name for the BLAS actually loaded into this process."""
    mkl = _mkl()
    if mkl is not None:
        version = mkl.version()
        # "Intel(R) oneAPI Math Kernel Library Version 2025.1-Product Build ..." -> 2025.1
        token = ""
        for word in version.split():
            if word[:1].isdigit() and "." in word:
                token = word.split("-")[0]
                break
        return "MKL " + token if token else "MKL"
    libs = _loaded_libraries()
    if "libopenblas" in libs:
        return "OpenBLAS"
    if "libblis" in libs:
        return "BLIS"
    return "unknown - assume serial"


def apply_budget() -> None:
    """Push the budget to the BLAS, **once**, and only when the budget is ours.

    Idempotent. A budget that came from ``OMP_NUM_THREADS`` or from the core count is left
    where the environment put it (module docstring); an explicit ``KUIVA_NUM_THREADS`` or
    :func:`set_budget` is what makes this one knob govern the BLAS as well as the kernels.
    """
    global _APPLIED
    if _APPLIED:
        return
    _APPLIED = True
    if budget_source() not in ("KUIVA_NUM_THREADS", "explicit"):
        return
    mkl = _mkl()
    if mkl is None:
        log.debug("thread budget %d not applied to the BLAS: %s exposes no thread control",
                  budget(), blas_identity())
        return
    mkl.set_global(budget())


# --- the regions --------------------------------------------------------------------------


class _Region(object):
    """Context manager for one thread policy; restores everything it changed."""

    __slots__ = ("_kernel_threads", "_blas_threads", "_saved_kernel", "_saved_blas")

    def __init__(self, kernel_threads: int, blas_threads: int):
        self._kernel_threads = kernel_threads
        self._blas_threads = blas_threads
        self._saved_kernel = None
        self._saved_blas = None

    def __enter__(self) -> int:
        global _REGION_KERNEL_THREADS
        self._saved_kernel = _REGION_KERNEL_THREADS
        _REGION_KERNEL_THREADS = self._kernel_threads
        mkl = _mkl()
        if mkl is not None:
            self._saved_blas = mkl.set_local(self._blas_threads)
        return self._kernel_threads

    def __exit__(self, *exc) -> bool:
        global _REGION_KERNEL_THREADS
        _REGION_KERNEL_THREADS = self._saved_kernel
        if self._saved_blas is not None:
            mkl = _mkl()
            if mkl is not None:
                mkl.set_local(self._saved_blas)
        return False


def blas_region(n_threads: Optional[int] = None) -> _Region:
    """A dense-linear-algebra stage: the BLAS gets the budget, kernels get one thread.

    Enter once per stage (integral transform, orbital optimizer, sigma build), never
    inside a loop. Yields the kernel thread count, which is 1 by construction.
    """
    n = budget() if n_threads is None else _positive(n_threads, "n_threads")
    apply_budget()
    return _Region(kernel_threads=1, blas_threads=n)


def kernel_region(n_threads: Optional[int] = None) -> _Region:
    """A kernel-bound stage: compiled kernels get the budget, the BLAS is clamped to one.

    Enter once per solve or sweep. Yields the kernel thread count, so a caller that wants
    to pass it explicitly can take it from the ``with`` statement instead of calling
    :func:`thread_count` again.

    ⚠ Clamping the BLAS is not a courtesy to the kernels — it is the measured policy of
    measured: inside this layer the threaded BLAS returns
    nothing for the CPU seconds it spends, and those seconds are what the port gate ranks candidates
    by. Nested kernels that dispatch a GEMM themselves clamp MKL again inside their own
    parallel region; the two clamps agree by construction.
    """
    n = budget() if n_threads is None else _positive(n_threads, "n_threads")
    apply_budget()
    return _Region(kernel_threads=n, blas_threads=1)


def blas_stage(function):
    """Decorator form of :func:`blas_region`, for a driver whose body is the stage.

    ⚠ Used where the alternative is re-indenting a long validated driver
    (:func:`kuiva.mcscf.orbopt.optimize_orbitals` is the case in point, and the region rule says
    that loop is extended, never restructured). A region entered by decorator is entered
    exactly once per call, which is the same guarantee the ``with`` form gives.
    """
    import functools

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        with blas_region():
            return function(*args, **kwargs)
    return wrapper


def kernel_stage(function):
    """Decorator form of :func:`kernel_region` (see :func:`blas_stage`)."""
    import functools

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        with kernel_region():
            return function(*args, **kwargs)
    return wrapper


# --- the probe ----------------------------------------------------------------------------


class ThreadsReport(NamedTuple):
    """What :func:`threads_report` measured. ``None`` fields mean "not measurable here"."""

    budget: int
    source: str
    blas: str
    blas_max_threads: Optional[int]
    speedup: Optional[float]
    serial_cpu_per_wall: Optional[float]
    threaded: Optional[bool]

    def summary(self) -> str:
        """The one-line banner statement (ASCII, no measured number)."""
        if self.threaded is None:
            # unmeasurable, and it says so: no thread control on this BLAS, or a budget
            # of one, where there is nothing to verify in the first place
            threaded = "" if self.budget == 1 else ", threading unverified"
        else:
            threaded = ", threaded" if self.threaded else ", NOT threaded"
        return "threads: {} ({}); BLAS: {}{}".format(self.budget, self.source, self.blas,
                                                     threaded)


def _time_gemm(a, b, n_threads: int):
    """One complex GEMM at a given MKL width; returns ``(wall, cpu)`` seconds."""
    mkl = _mkl()
    saved = None if mkl is None else mkl.set_local(n_threads)
    try:
        w0, c0 = time.perf_counter(), time.process_time()
        a.dot(b)
        return time.perf_counter() - w0, time.process_time() - c0
    finally:
        if saved is not None and mkl is not None:
            mkl.set_local(saved)


def _measure(n_threads: int):
    """Time one fixed GEMM at one thread and at ``n_threads``; return their ratio.

    Returns ``(speedup, serial_cpu_per_wall)``, or ``(None, None)`` when there is nothing
    to measure. The whole probe costs ~40 ms of identical work on every run, which is what
    lets it run unconditionally at every banner (:data:`_PROBE_DIM`).

    ⚠ **Without thread control there is no measurement, and the honest answer is
    ``None``** — not a speedup of 1. Both calls would run at the same ambient width, so
    the ratio would say "not threaded" about a BLAS that may be threading perfectly, and
    the warning would fire on every run of a machine whose only fault is not using MKL.
    Degrading to a no-op is allowed; pretending to have measured is not.
    """
    if _mkl() is None:
        log.debug("no BLAS thread control (%s): the thread budget cannot be verified",
                  blas_identity())
        return None, None
    try:
        import numpy as np
    except ImportError:                                   # pragma: no cover - numpy is core
        return None, None
    rng = np.random.default_rng(0)
    a = (rng.standard_normal((_PROBE_DIM, _PROBE_DIM))
         + 1j * rng.standard_normal((_PROBE_DIM, _PROBE_DIM)))
    b = a.copy()
    _time_gemm(a, b, 1)                                   # warm up MKL's dispatch and pages
    # Best of `_PROBE_REPEATS`, both sides: a millisecond-scale timing on a machine that is
    # also running everything else has a long tail upward and none downward, so the minimum
    # is the estimator with the least noise in it — and the ratio of two minima is what the
    # ratio-not-absolute rule wants. The count is fixed, so the work is identical
    # on every run (see _PROBE_DIM).
    wall_1, cpu_1 = min((_time_gemm(a, b, 1) for _ in range(_PROBE_REPEATS)),
                        key=lambda t: t[0])
    if wall_1 <= 0.0:                                     # pragma: no cover - clock granularity
        return None, None
    if n_threads == 1:
        return 1.0, cpu_1 / wall_1
    _time_gemm(a, b, n_threads)
    wall_n = min(_time_gemm(a, b, n_threads)[0] for _ in range(_PROBE_REPEATS))
    if wall_n <= 0.0:                                     # pragma: no cover
        return None, cpu_1 / wall_1
    return wall_1 / wall_n, cpu_1 / wall_1


def threads_report(force: bool = False) -> ThreadsReport:
    """Identify the BLAS, measure whether it threads, warn when it does not — once.

    Cached per process: the banner, a driver and the standalone entry point all get the
    same measurement, and no run pays for it twice.
    """
    global _REPORT
    if _REPORT is not None and not force:
        return _REPORT
    apply_budget()
    n = budget()
    mkl = _mkl()
    speedup, cpu_per_wall = _measure(n)
    threaded = None if speedup is None or n == 1 else speedup >= THREADED_SPEEDUP_MIN
    report = ThreadsReport(budget=n, source=budget_source(), blas=blas_identity(),
                           blas_max_threads=None if mkl is None else mkl.max_threads(),
                           speedup=speedup, serial_cpu_per_wall=cpu_per_wall,
                           threaded=threaded)
    _REPORT = report
    _warn_about(report)
    return report


def _warn_about(report: ThreadsReport) -> None:
    """The two failure modes that are silent by nature (module docstring)."""
    if report.threaded is False:
        log.warning(
            "a thread budget of %d is configured, but the loaded BLAS (%s) showed no "
            "speedup when asked for %d threads (measured %.2fx). Threaded work is being "
            "charged CPU seconds it does not get back: check MKL_NUM_THREADS / "
            "OMP_NUM_THREADS, or set KUIVA_NUM_THREADS=1 and stop paying for it.",
            report.budget, report.blas, report.budget, report.speedup)
    if (report.serial_cpu_per_wall is not None
            and report.serial_cpu_per_wall > SERIAL_CPU_PER_WALL_MAX):
        log.warning(
            "cpu/wall is %.1f for a BLAS call asked to run on ONE thread: a second "
            "threading runtime is spinning next to it (a libgomp build loaded beside "
            "libiomp5), or KMP_BLOCKTIME is leaving idle teams spinning. Every "
            "cpu_seconds figure taken in this process is inflated until it is fixed "
            "(set KMP_BLOCKTIME=0 for profiling runs).",
            report.serial_cpu_per_wall)


def banner_entry() -> str:
    """The run banner's thread line — one deterministic sentence, no measured number.

    The measured ratio itself goes to DEBUG and to :func:`threads_report`: a number that
    moves by a few percent between runs does not belong in an output file that is diffed
    against a committed reference. What the banner states is the *verdict*, which is
    stable on a given machine, and the warnings above carry the numbers when they matter.
    """
    report = threads_report()
    log.debug("thread probe: budget %d (%s), BLAS %s, speedup %s, serial cpu/wall %s",
              report.budget, report.source, report.blas,
              "n/a" if report.speedup is None else "{:.2f}".format(report.speedup),
              "n/a" if report.serial_cpu_per_wall is None
              else "{:.2f}".format(report.serial_cpu_per_wall))
    return report.summary()


def _reset_cache() -> None:
    """Testing hook: forget the resolved budget, the applied flag and the measurement."""
    global _BUDGET, _SOURCE, _APPLIED, _REPORT, _REGION_KERNEL_THREADS
    _BUDGET = _SOURCE = _REPORT = None
    _APPLIED = False
    _REGION_KERNEL_THREADS = None


def main() -> None:
    """``python -m kuiva.util.threads`` — the standalone probe, with its numbers."""
    from . import output as out

    report = threads_report()
    out.section(log, "Thread policy", clock=False)
    out.entries(log, [
        ("thread budget", report.budget, "", report.source),
        ("BLAS in this process", report.blas),
        ("BLAS max threads", report.blas_max_threads),
        ("measured speedup at the budget",
         None if report.speedup is None else report.speedup, "", "", "{:.2f}"),
        ("cpu/wall at one thread",
         None if report.serial_cpu_per_wall is None else report.serial_cpu_per_wall,
         "", "", "{:.2f}"),
        ("threaded BLAS", report.threaded),
    ])
    out.blank(log)
    out.note(log, "regions: blas_region() gives the BLAS the budget; kernel_region()")
    out.note(log, "gives it to the compiled kernels and clamps the BLAS to one thread.")


__all__ = ["THREADED_SPEEDUP_MIN", "SERIAL_CPU_PER_WALL_MAX", "ThreadsReport",
           "budget", "budget_source", "set_budget", "thread_count", "apply_budget",
           "blas_identity", "blas_region", "kernel_region", "blas_stage", "kernel_stage",
           "threads_report", "banner_entry", "main"]


if __name__ == "__main__":                                # pragma: no cover - entry point
    main()
