"""Thread control: the knob, the two regions, and the probe's two warnings.

What this file is for (and what it deliberately leaves to the machine): the *policy* is
tested exhaustively and deterministically — precedence, region behaviour, nesting,
restoration, and both warnings driven by injected measurements — while the *measurement*
is asserted only to be sane and self-consistent on whatever BLAS is loaded. A test that
demanded a particular speedup would be asserting the thermal state of the development box
, which is exactly what the probe reports a ratio to avoid.
"""
import numpy as np
import pytest

from kuiva.ci import kernels
from kuiva.ci import strings
from kuiva.ci.strings import Determinants, connections
from kuiva.util import threads


@pytest.fixture(autouse=True)
def fresh_threads():
    """Every test starts from an unresolved budget and no region, and leaves one behind.

    ⚠ It also restores the BLAS's **global** width. Several tests here call
    :func:`~kuiva.util.threads.set_budget`, which by design pushes the budget to the BLAS
    for the whole process — so without this, a test that asked for two threads would leave
    every later test in the session running at two, and the suite's behaviour would
    depend on its order.
    """
    control = threads._blas_control()
    restore = None if control is None else control.max_threads()
    threads._reset_cache()
    try:
        yield
    finally:
        threads._reset_cache()
        if restore is not None:
            control.set_global(restore)


# --- the knob ------------------------------------------------------------------------------

def test_precedence_of_the_budget(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    monkeypatch.setenv("KUIVA_NUM_THREADS", "2")
    assert threads.budget() == 2 and threads.budget_source() == "KUIVA_NUM_THREADS"
    assert threads.thread_count() == 2
    assert threads.thread_count(5) == 5                  # explicit beats everything

    threads._reset_cache()
    monkeypatch.delenv("KUIVA_NUM_THREADS")
    assert threads.budget() == 3 and threads.budget_source() == "OMP_NUM_THREADS"

    threads._reset_cache()
    monkeypatch.delenv("OMP_NUM_THREADS")
    assert threads.budget() == threads._all_cores()
    assert threads.budget_source() == "all cores"


def test_the_core_count_is_the_affinity_mask_not_the_machine():
    """A cgroup or a taskset is a statement about this process, and it is respected."""
    import os

    if not hasattr(os, "sched_getaffinity"):             # pragma: no cover - non-Linux
        pytest.skip("no sched_getaffinity on this platform")
    assert threads._all_cores() == len(os.sched_getaffinity(0))
    assert threads._all_cores() <= (os.cpu_count() or 1)


def test_a_nonsense_budget_is_refused(monkeypatch):
    with pytest.raises(ValueError, match="positive"):
        threads.thread_count(0)
    with pytest.raises(ValueError, match="positive"):
        threads.set_budget(-4)
    monkeypatch.setenv("KUIVA_NUM_THREADS", "zero")
    with pytest.raises(ValueError, match="KUIVA_NUM_THREADS"):
        threads.budget()


def test_set_budget_is_the_top_of_the_chain(monkeypatch):
    monkeypatch.setenv("KUIVA_NUM_THREADS", "2")
    assert threads.set_budget(3) == 3
    assert threads.budget() == 3 and threads.budget_source() == "explicit"


# --- the regions ---------------------------------------------------------------------------

def test_outside_every_region_a_kernel_gets_the_whole_budget():
    """The pre-region behaviour, kept deliberately: since the compiled backend landed, an
    un-regioned path is a threaded-kernel path, and BLAS-region-by-default would have
    silently serialized it."""
    threads.set_budget(4)
    assert threads.thread_count() == 4


def test_blas_region_gives_the_kernels_one_thread():
    threads.set_budget(4)
    with threads.blas_region() as n:
        assert n == 1
        assert threads.thread_count() == 1
        assert threads.thread_count(2) == 2               # explicit still wins
    assert threads.thread_count() == 4                    # restored


def test_kernel_region_gives_the_kernels_the_budget():
    threads.set_budget(4)
    with threads.kernel_region() as n:
        assert n == 4 and threads.thread_count() == 4
    assert threads.thread_count() == 4


def test_regions_nest_and_restore_in_order():
    """The real shape: an orbital optimizer's BLAS region containing a DMRG solver's
    kernel region containing a determinant scan's."""
    threads.set_budget(3)
    with threads.blas_region():
        assert threads.thread_count() == 1
        with threads.kernel_region():
            assert threads.thread_count() == 3
            with threads.blas_region():
                assert threads.thread_count() == 1
            assert threads.thread_count() == 3
        assert threads.thread_count() == 1
    assert threads.thread_count() == 3


def test_a_region_restores_even_when_the_body_raises():
    threads.set_budget(2)
    with pytest.raises(RuntimeError):
        with threads.blas_region():
            raise RuntimeError("boom")
    assert threads.thread_count() == 2


def test_an_explicit_region_width_overrides_the_budget():
    threads.set_budget(4)
    with threads.kernel_region(2) as n:
        assert n == 2 and threads.thread_count() == 2


def test_the_stage_decorators_are_the_regions():
    threads.set_budget(4)

    @threads.blas_stage
    def dense():
        return threads.thread_count()

    @threads.kernel_stage
    def kernelly():
        return threads.thread_count()

    assert dense() == 1
    assert kernelly() == 4
    assert threads.thread_count() == 4


# --- the BLAS control, whichever library is loaded -------------------------------------------

def _control_or_skip():
    """The bound thread control, or skip. ⚠ Not "the MKL": every test below is written
    against the *interface*, so the same assertions run on OpenBLAS and BLIS — which is the
    point of there being one interface, and the only way a machine that is not this one
    checks its own dispatch."""
    control = threads._blas_control()
    if control is None:
        pytest.skip("the loaded BLAS ({}) exposes no thread control"
                    .format(threads.blas_identity()))
    return control


def test_the_blas_is_identified_from_the_process_not_from_a_build_flag():
    """Whatever it is, the identity is a non-empty statement and matches the maps."""
    name = threads.blas_identity()
    assert name and name == threads.blas_identity()       # stable within a process
    libs = threads._loaded_libraries()
    if name.startswith("MKL"):
        assert "mkl" in libs
    elif name.startswith("OpenBLAS"):                     # pragma: no cover - not this box
        assert "openblas" in libs
    elif name.startswith("BLIS"):                         # pragma: no cover - not this box
        assert "libblis" in libs


def test_kernel_region_clamps_the_blas_and_gives_it_back():
    control = _control_or_skip()
    threads.set_budget(4)
    before = control.max_threads()
    with threads.kernel_region():
        assert control.max_threads() == 1
    assert control.max_threads() == before                # exactly what it was


def test_blas_region_hands_the_blas_the_budget():
    control = _control_or_skip()
    threads.set_budget(2)
    with threads.blas_region():
        assert control.max_threads() == 2


def test_the_budget_reaches_the_blas_only_when_it_is_ours(monkeypatch):
    """⚠ An OMP_NUM_THREADS budget leaves the BLAS where the environment put it —
    silently overriding a deliberate MKL_NUM_THREADS would be the same class of surprise
    this module exists to remove."""
    control = _control_or_skip()
    ambient = control.max_threads()
    try:
        monkeypatch.delenv("KUIVA_NUM_THREADS", raising=False)
        monkeypatch.setenv("OMP_NUM_THREADS", str(max(1, ambient - 1)))
        threads.apply_budget()
        assert control.max_threads() == ambient           # untouched

        threads._reset_cache()
        monkeypatch.setenv("KUIVA_NUM_THREADS", "1")
        threads.apply_budget()
        assert control.max_threads() == 1
    finally:
        # this one test writes a process-global BLAS setting; a failure in the middle of
        # it must not leave every later test running single-threaded
        control.set_global(ambient)


# --- the dispatch: which setter a given library gets ------------------------------------------
#
# ⚠ Neither OpenBLAS nor BLIS can be exercised on the development box, and a control layer
# nobody can run is a control layer that silently does not work. What *is* testable here is
# the dispatch — which library is picked out of the process map, which symbols are bound to
# it, and which of them a region calls — so that is asserted against a faked library. The
# measured verdict itself is confirmed once on the target machine.


class _FakeSymbol(object):
    """A ctypes-like function: settable ``restype``/``argtypes``, and it records its calls."""

    def __init__(self, name, calls, result=0):
        self.name, self._calls, self._result = name, calls, result
        self.restype = None
        self.argtypes = None

    def __call__(self, *args):
        self._calls.append((self.name,) + args)
        return self._result


class _FakeLib(object):
    """A shared library exporting exactly ``symbols`` and nothing else.

    The "nothing else" half is the point: binding falls through to the next candidate name
    when a symbol is missing, and an export list that answered every request would make
    every fallback path in :func:`~kuiva.util.threads._first_symbol` untestable.
    """

    def __init__(self, symbols, calls):
        self._symbols, self._calls = dict(symbols), calls

    def __getattr__(self, name):
        if name not in self._symbols:
            raise AttributeError(name)
        symbol = _FakeSymbol(name, self._calls, self._symbols[name])
        self.__dict__[name] = symbol                      # ctypes caches its funcptrs too
        return symbol


def _fake_process(monkeypatch, path, symbols):
    """Pretend ``path`` is mapped into this process and exports ``symbols``."""
    calls = []
    lib = _FakeLib(symbols, calls)
    monkeypatch.setattr(threads, "_ensure_blas_loaded", lambda: None)
    monkeypatch.setattr(threads, "_mapped_library_paths",
                        lambda token: [path] if token in path.lower() else [])
    # ⚠ The identity falls back to the raw map when nothing binds, so the fake process has
    # to own that too — otherwise the fallback reads the *real* one and answers about the
    # machine the test is running on.
    monkeypatch.setattr(threads, "_loaded_libraries", lambda: path.lower())
    monkeypatch.setattr(threads.ctypes, "CDLL",
                        lambda name, *a, **k: lib if name == path else _raise_oserror(name))
    monkeypatch.setattr(threads, "_BLAS_HANDLE", None, raising=False)
    monkeypatch.setattr(threads, "_BLAS_LOOKED", False, raising=False)
    return calls


def _raise_oserror(name):
    raise OSError("no such library in this fake process: {}".format(name))


def test_a_wheel_openblas_is_bound_by_path_and_gets_openblas_calls(monkeypatch):
    """⚠ By *path*, not by soname: NumPy ships its OpenBLAS as
    ``libopenblas64_-r0-<hash>.so`` inside ``numpy.libs``, which no ``dlopen`` of
    ``libopenblas.so.0`` will ever find — and the 64-bit build renames its exports, so the
    plain symbol names are not there either."""
    calls = _fake_process(
        monkeypatch, "/wheel/numpy.libs/libopenblas64_-r0-abc123.so",
        {"openblas_set_num_threads64_": None, "openblas_get_num_threads64_": 7,
         "openblas_get_num_procs64_": 16, "openblas_get_config64_":
             b"OpenBLAS 0.3.27  USE64BITINT DYNAMIC_ARCH"})
    control = threads._blas_control()
    assert isinstance(control, threads._OpenBLAS)
    assert control.name == "OpenBLAS 0.3.27"
    assert control.thread_local is False                  # one process-wide width
    assert control.max_threads() == 7
    control.set_global(3)
    assert ("openblas_set_num_threads64_", 3) in calls
    assert not any(name.startswith("MKL") or name.startswith("bli_") for name, *_ in calls)


def test_a_region_saves_and_restores_through_the_openblas_getter(monkeypatch):
    """The region contract on a library with no thread-local setting: it must read the
    width, set its own, and put the read-back value there again on the way out."""
    calls = _fake_process(
        monkeypatch, "/usr/lib/libopenblas.so.0",
        {"openblas_set_num_threads": None, "openblas_get_num_threads": 5,
         "openblas_get_num_procs": 16})
    threads.set_budget(2)
    with threads.kernel_region():
        pass
    widths = [args[0] for name, *args in calls if name == "openblas_set_num_threads"]
    assert widths[-2:] == [1, 5]                          # clamped, then given back exactly


def test_blis_is_bound_to_its_own_entry_points(monkeypatch):
    calls = _fake_process(monkeypatch, "/opt/blis/lib/libblis.so.4",
                          {"bli_thread_set_num_threads": None,
                           "bli_thread_get_num_threads": 9,
                           "bli_info_get_version_str": b"0.9.0"})
    control = threads._blas_control()
    assert isinstance(control, threads._BLIS)
    assert control.name == "BLIS 0.9.0" and control.thread_local is False
    assert control.max_threads() == 9
    control.set_global(4)
    assert ("bli_thread_set_num_threads", 4) in calls


def test_an_openblas_without_a_getter_shadows_the_width_it_set(monkeypatch):
    """⚠ Older builds export no ``openblas_get_num_threads``. Refusing to control the BLAS
    at all would be worse than a shadow that is wrong only if something else in the process
    moves the width, so the control keeps its own value — seeded from the processor count,
    which is where OpenBLAS itself starts."""
    _fake_process(monkeypatch, "/usr/lib/libopenblas.so.0",
                  {"openblas_set_num_threads": None, "openblas_get_num_procs": 12})
    control = threads._blas_control()
    assert control.max_threads() == 12                    # the seed
    assert control.set_local(3) == 12 and control.max_threads() == 3


def test_a_blas_with_no_thread_control_at_all_binds_nothing(monkeypatch):
    """A library that is there and exposes no setter is not a control: the probe must then
    report "unverified" rather than bind half an interface."""
    _fake_process(monkeypatch, "/usr/lib/libopenblas.so.0", {"openblas_get_num_procs": 8})
    assert threads._blas_control() is None
    assert threads.blas_identity() == "OpenBLAS"          # still identified, from the maps


def test_mkl_wins_when_two_blas_libraries_are_mapped(monkeypatch):
    """A process can carry both — PySCF's MKL and a wheel's OpenBLAS — and the one that
    runs Kuiva's GEMMs is the one NumPy linked. ⚠ MKL first is also what keeps this box's
    answer identical to what it was before the other two libraries were supported."""
    calls = []
    mkl = _FakeLib({"MKL_Set_Num_Threads_Local": 0, "MKL_Set_Num_Threads": None,
                    "MKL_Get_Max_Threads": 4, "MKL_Get_Version_String": None}, calls)
    openblas = _FakeLib({"openblas_set_num_threads": None,
                         "openblas_get_num_threads": 8}, calls)
    libs = {"libmkl_rt.so.2": mkl, "/wheel/libopenblas64_.so": openblas}
    mapped = ["/opt/intel/libmkl_rt.so.2", "/wheel/libopenblas64_.so"]
    monkeypatch.setattr(threads, "_ensure_blas_loaded", lambda: None)
    monkeypatch.setattr(threads, "_mapped_library_paths",
                        lambda token: [p for p in mapped if token in p.lower()])
    monkeypatch.setattr(threads.ctypes, "CDLL",
                        lambda name, *a, **k: libs.get(name) or _raise_oserror(name))
    monkeypatch.setattr(threads, "_BLAS_HANDLE", None, raising=False)
    monkeypatch.setattr(threads, "_BLAS_LOOKED", False, raising=False)
    assert isinstance(threads._blas_control(), threads._MKL)


# --- the probe: the two warnings that are otherwise silent ---------------------------------

class _FakeControl(object):
    """Just enough of the interface to stand in for a BLAS whose width can be set."""

    thread_local = True
    name = "MKL 2025.1"

    def __init__(self, max_threads=4):
        self._max = max_threads

    def version(self):
        return "Intel(R) oneAPI Math Kernel Library Version 2025.1-Product Build 0"

    def max_threads(self):
        return self._max

    def set_local(self, n):
        previous, self._max = self._max, n
        return previous

    def set_global(self, n):
        self._max = n


def test_the_probe_warns_when_the_budget_buys_no_threads(monkeypatch, kuiva_caplog):
    """"You think it is threaded and it is not" — the case that silently wastes CPU-hours."""
    threads.set_budget(4)
    monkeypatch.setattr(threads, "_blas_control", lambda: _FakeControl())      # thread control exists
    monkeypatch.setattr(threads, "_measure", lambda n: (1.02, 1.0))
    monkeypatch.setattr(threads, "blas_identity", lambda: "MKL 2025.1")
    report = threads.threads_report(force=True)
    assert report.threaded is False
    warnings = [r.getMessage() for r in kuiva_caplog.records if r.levelname == "WARNING"]
    assert any("no speedup" in m and "KUIVA_NUM_THREADS" in m for m in warnings)
    assert ", NOT threaded" in report.summary()


def test_the_probe_warns_when_a_second_runtime_is_spinning(monkeypatch, kuiva_caplog):
    """The other direction: cpu/wall >> 1 at nominal one thread corrupts every cpu_seconds
    figure taken afterwards."""
    threads.set_budget(4)
    monkeypatch.setattr(threads, "_blas_control", lambda: _FakeControl())
    monkeypatch.setattr(threads, "_measure", lambda n: (3.8, 4.0))
    report = threads.threads_report(force=True)
    assert report.threaded is True                        # the speedup itself is fine
    warnings = [r.getMessage() for r in kuiva_caplog.records if r.levelname == "WARNING"]
    assert any("cpu/wall" in m and "KMP_BLOCKTIME" in m for m in warnings)


def test_a_healthy_probe_says_nothing(monkeypatch, kuiva_caplog):
    threads.set_budget(4)
    monkeypatch.setattr(threads, "_blas_control", lambda: _FakeControl())
    monkeypatch.setattr(threads, "_measure", lambda n: (3.7, 1.02))
    report = threads.threads_report(force=True)
    assert report.threaded is True
    assert not [r for r in kuiva_caplog.records if r.levelname == "WARNING"]


def test_without_thread_control_the_probe_reports_unverified_rather_than_unthreaded(
        monkeypatch, kuiva_caplog):
    """⚠ Degrading to a no-op is allowed; pretending to have measured is not. Two calls at
    the same ambient width would show a speedup of 1 and accuse a perfectly threaded BLAS."""
    threads.set_budget(4)
    monkeypatch.setattr(threads, "_blas_control", lambda: None)
    monkeypatch.setattr(threads, "blas_identity", lambda: "OpenBLAS")
    report = threads.threads_report(force=True)
    assert report.speedup is None and report.threaded is None
    assert "unverified" in report.summary()
    assert not [r for r in kuiva_caplog.records if r.levelname == "WARNING"]


class _InertControl(object):
    """A BLAS that accepts a width and keeps its own — the OpenMP-built OpenBLAS's setter."""

    thread_local = False
    name = "OpenBLAS 0.3.27"

    def set_local(self, n):
        return 8

    def set_global(self, n):
        pass

    def max_threads(self):
        return 8


def test_a_setter_that_does_not_take_reports_unverified_rather_than_accusing(
        monkeypatch, kuiva_caplog):
    """⚠ The false accusation this exists to prevent: with an inert setter both probe GEMMs
    run at the same ambient width, so the measured speedup is 1 and a perfectly threaded
    BLAS would be reported as NOT threaded — on every run, on a machine whose only fault is
    an OpenMP-built OpenBLAS. The honest answer is that nothing could be measured."""
    threads.set_budget(4)
    monkeypatch.setattr(threads, "_blas_control", lambda: _InertControl())
    report = threads.threads_report(force=True)
    assert report.speedup is None and report.threaded is None
    assert "unverified" in report.summary()
    assert not [r for r in kuiva_caplog.records if r.levelname == "WARNING"]


def test_a_budget_of_one_has_nothing_to_verify(monkeypatch):
    threads.set_budget(1)
    monkeypatch.setattr(threads, "_blas_control", lambda: _FakeControl())
    monkeypatch.setattr(threads, "_measure", lambda n: (1.0, 1.0))
    report = threads.threads_report(force=True)
    assert report.threaded is None and "threaded" not in report.summary()


def test_the_report_is_measured_once_per_process(monkeypatch):
    threads.set_budget(2)
    calls = []
    monkeypatch.setattr(threads, "_blas_control", lambda: _FakeControl())
    monkeypatch.setattr(threads, "_measure", lambda n: (calls.append(n), (2.0, 1.0))[1])
    threads.threads_report()
    threads.threads_report()
    assert calls == [2]


def test_the_real_probe_is_self_consistent():
    """On this machine, whatever it is: the report describes what was actually measured."""
    report = threads.threads_report(force=True)
    assert report.budget >= 1
    assert report.blas
    if report.speedup is None:
        assert report.threaded is None                    # no control -> no claim
    else:
        assert report.speedup > 0.0 and report.serial_cpu_per_wall > 0.0
        if report.budget > 1:
            assert report.threaded == (report.speedup >= threads.THREADED_SPEEDUP_MIN)


def test_the_banner_line_carries_no_measured_number():
    """A number that moves a few percent between runs does not belong in an output file
    that is diffed against a committed reference."""
    line = threads.banner_entry()
    assert line.startswith("threads: ")
    assert threads.blas_identity() in line
    assert "x" not in line.split("BLAS:")[0]              # no "3.7x" in the budget half
    assert line == line.encode("ascii", "ignore").decode()   # ASCII only


# --- the policy where it is actually consumed ----------------------------------------------

def _dets(n=120, n_spinor=16, n_elec=8):
    rng = np.random.default_rng(3)
    seen, masks = set(), []
    while len(masks) < n:
        m = 0
        for p in rng.choice(n_spinor, size=n_elec, replace=False):
            m |= 1 << int(p)
        if m not in seen:
            seen.add(m)
            masks.append(m)
    return Determinants(np.array(masks, dtype=np.uint64), n_spinor, n_elec)


def test_the_determinant_scan_keeps_its_threads_inside_a_blas_region(monkeypatch):
    """⚠ The scan is kernel-bound and contains no BLAS at all, so it enters its own kernel
    region: a caller that happens to be a dense stage may not serialize it."""
    seen = []
    reference = kernels.resolve("connections_scan", "numpy")

    def spy(*args):
        seen.append(args[-1])
        return reference(*args)

    monkeypatch.setattr(strings.kernels, "resolve", lambda *a, **k: spy)
    threads.set_budget(3)
    with threads.blas_region():
        connections(_dets())
    assert seen and set(seen) == {3}


def test_an_explicit_thread_count_still_wins_in_the_scan(monkeypatch):
    seen = []
    reference = kernels.resolve("connections_scan", "numpy")

    def spy(*args):
        seen.append(args[-1])
        return reference(*args)

    monkeypatch.setattr(strings.kernels, "resolve", lambda *a, **k: spy)
    threads.set_budget(4)
    connections(_dets(), n_threads=1)
    assert set(seen) == {1}
