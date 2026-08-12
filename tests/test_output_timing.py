"""Tests for the output grammar and the timers.

These are not decoration. The INFO stream is the program's output file and requires
every iterative procedure to use the same table; if that drifts, every later module's output
drifts with it and the log stops being interpretable across subsystems. On the dev box only CPU time is
the cost measure, so a timer that recorded only wall time would actively mislead the profiling
porting decisions.
"""
import logging
import time

import pytest

from kuiva.util import output as out
from kuiva.util.logging import KuivaFormatter, get_logger
from kuiva.util.timing import REGISTRY, Timer, TimingRegistry, reset, summary, timed, timer


class Capture(logging.Handler):
    """Collect formatted records, so a test can look at the output as the user sees it."""

    def __init__(self, formatter=None):
        super().__init__()
        self.lines = []
        self.setFormatter(formatter or KuivaFormatter())

    def emit(self, record):
        self.lines.append(self.format(record))


@pytest.fixture
def cap():
    log = logging.getLogger("kuiva.test_output")
    handler = Capture()
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)
    log.propagate = False
    try:
        yield log, handler
    finally:
        log.removeHandler(handler)


# --- The formatter policy ----------------------------------------------------------------
def test_info_is_printed_verbatim(cap):
    """INFO *is* the output file: a prefix would destroy every table's alignment."""
    log, h = cap
    log.info("   label                    value")
    assert h.lines == ["   label                    value"]


def test_warnings_are_marked_and_attributed(cap):
    """Greppable, and it names the subsystem — a warning 2000 lines into an output file is
    useless if you cannot tell which module raised it."""
    log, h = cap
    log.warning("2 vectors dropped")
    assert h.lines[0].startswith(" *** WARNING [test_output] ")
    assert "2 vectors dropped" in h.lines[0]


def test_errors_are_marked(cap):
    log, h = cap
    log.error("did not converge")
    assert " *** ERROR [test_output] " in h.lines[0]


def test_debug_keeps_a_prefix(cap):
    log, h = cap
    log.debug("block dims 4 x 4")
    assert h.lines[0].startswith("[") and "DEBUG" in h.lines[0]


def test_multiline_warning_is_indented(cap):
    log, h = cap
    log.warning("first line\nsecond line")
    assert h.lines[0].splitlines()[1].startswith("     second")


# --- The grammar -------------------------------------------------------------------------
def test_entry_layout_is_fixed(cap):
    log, h = cap
    out.entry(log, "AO basis functions", 116)
    line = h.lines[0]
    assert line.startswith(out.INDENT)
    assert line.endswith("116")
    assert len(line) == len(out.INDENT) + out.LABEL_W + out.VALUE_W


def test_entry_units_and_notes(cap):
    log, h = cap
    out.entry(log, "SCF energy", -100.0178158485, "Eh", fmt=out.E_FMT)
    out.entry(log, "working-basis functions", 114, note="2 dropped")
    assert "-100.017815848500  Eh" in h.lines[0]
    assert h.lines[1].endswith("(2 dropped)")


def test_booleans_and_none_are_readable(cap):
    log, h = cap
    out.entry(log, "converged", True)
    out.entry(log, "auxiliary basis", None)
    assert h.lines[0].endswith("yes")
    assert h.lines[1].endswith("-")


def test_section_and_subsection_rules(cap):
    log, h = cap
    out.section(log, "Integral transformation")
    assert h.lines[1] == "=" * out.WIDTH
    assert h.lines[2].startswith(" Integral transformation") and "t =" in h.lines[2]
    assert len(h.lines[2]) <= out.WIDTH + 4
    h.lines.clear()
    out.subsection(log, "Cholesky decomposition")
    assert h.lines[1].startswith(" -- Cholesky decomposition ") and h.lines[1].endswith("-")


def test_table_header_row_footer(cap):
    """The output-grammar requirement, verified as a shape: header, rule, rows, rule."""
    log, h = cap
    t = out.Table(log, [out.col_iter(), out.col_energy(), out.col_delta(), out.col_time()])
    t.start()
    t.row(1, -100.5, 1.2e-3, 0.5)
    t.row(2, -100.6, 1.0e-6, 0.4)
    t.end("converged in 2 iterations")
    assert "iter" in h.lines[0] and "energy [Eh]" in h.lines[0]
    assert set(h.lines[1].strip()) == {"-", " "}
    assert "-100.500000000000" in h.lines[2] and "1.200e-03" in h.lines[2]
    assert h.lines[4] == h.lines[1]                     # closing rule matches the opening one
    assert h.lines[5].strip() == "converged in 2 iterations"
    assert t.n_rows == 2
    # every row lines up with the header, which is the whole point of a fixed grammar
    assert len({len(x) for x in h.lines[:5]}) == 1


def test_table_rejects_a_wrong_row_width(cap):
    log, _ = cap
    t = out.Table(log, [out.col_iter(), out.col_energy()])
    t.start()
    with pytest.raises(ValueError, match="2 columns"):
        t.row(1)


def test_column_never_narrower_than_its_header():
    c = out.Column("a-very-long-header", "{:.2f}", 4)
    assert len(c.render(1.0)) == len("a-very-long-header")


def test_matrix_dump_is_not_output(cap):
    """Matrices belong in the property dump, never in the log at INFO."""
    import numpy as np
    log, h = cap
    log.setLevel(logging.INFO)
    out.matrix(log, "h", np.eye(3))
    assert h.lines == []


# --- Timers ------------------------------------------------------------------------------
def test_timer_records_wall_and_cpu():
    reg = TimingRegistry()
    with Timer("spin", registry=reg) as t:
        x = 0
        for _ in range(200000):
            x += 1
    assert t.wall > 0.0 and t.cpu > 0.0
    node = reg.get("spin")
    assert node.calls == 1 and node.wall == pytest.approx(t.wall)


def test_cpu_and_wall_are_two_independent_clocks(monkeypatch):
    """Wall and CPU are separate measurements, and a region can consume a
    lot of one and little of the other.

    Driven by fake clocks rather than a real ``sleep``. A real sleep gives the wrong answer
    here for an interesting reason worth keeping in the suite's memory: ``process_time``
    counts CPU over *all* threads of the process, and after any threaded BLAS call the
    MKL/OpenMP worker threads keep spinning for ``KMP_BLOCKTIME`` (200 ms by default). A short
    region that merely follows a parallel one is therefore charged for other threads' idling
    — measured here at 0.35 s of "CPU" for a 0.05 s sleep. See the caveat in
    ``kuiva.util.timing``; it is a property of the runtime, not of this timer.
    """
    import kuiva.util.timing as tm

    walls = iter([100.0, 100.5])
    cpus = iter([10.0, 10.01])
    monkeypatch.setattr(tm.time, "perf_counter", lambda: next(walls))
    monkeypatch.setattr(tm.time, "process_time", lambda: next(cpus))
    reg = TimingRegistry()
    with Timer("sleep", registry=reg) as t:
        pass
    assert t.wall == pytest.approx(0.5)
    assert t.cpu == pytest.approx(0.01)
    assert reg.get("sleep").parallel_ratio == pytest.approx(0.02)


def test_a_busy_region_records_real_cpu():
    """Real clocks are wired up. Deliberately no assertion on ``cpu/wall``: in a process that
    has run threaded BLAS, spin-waiting OpenMP workers inflate it (see the test above)."""
    reg = TimingRegistry()
    with Timer("spin", registry=reg) as t:
        x = 0
        for _ in range(300000):
            x += 1
    assert t.cpu > 0.0 and t.wall > 0.0
    assert reg.get("spin").wall == pytest.approx(t.wall)


def test_nesting_builds_a_path():
    reg = TimingRegistry()
    with Timer("casscf", registry=reg):
        with Timer("ci", registry=reg):
            with Timer("sigma", registry=reg):
                pass
        with Timer("orbopt", registry=reg):
            pass
    paths = [n.path for n in reg.nodes()]
    assert paths == ["casscf", "casscf/ci", "casscf/ci/sigma", "casscf/orbopt"]
    assert [n.depth for n in reg.nodes()] == [0, 1, 2, 1]
    # the same kernel from two places is reported separately
    assert reg.get("casscf/ci/sigma") is not None and reg.get("sigma") is None


def test_repeated_calls_accumulate():
    reg = TimingRegistry()
    for _ in range(5):
        with Timer("kernel", registry=reg):
            pass
    assert reg.get("kernel").calls == 5


def test_exception_does_not_lose_the_measurement():
    reg = TimingRegistry()
    with pytest.raises(RuntimeError):
        with Timer("boom", registry=reg):
            raise RuntimeError("kaboom")
    assert reg.get("boom").calls == 1
    assert reg._stack == []                            # and the nesting stack is unwound


def test_decorator():
    reg = TimingRegistry()

    @timed("decorated", registry=reg)
    def f(a):
        return a * 2

    assert f(3) == 6
    assert reg.get("decorated").calls == 1


def test_totals_count_top_level_regions_only():
    """Nested times are inclusive, so summing everything would double count."""
    reg = TimingRegistry()
    with Timer("outer", registry=reg):
        with Timer("inner", registry=reg):
            time.sleep(0.01)
    assert reg.total_wall() == pytest.approx(reg.get("outer").wall)


def test_summary_emits_a_table(cap):
    log, h = cap
    reg = TimingRegistry()
    with Timer("step", registry=reg):
        pass
    summary(log, registry=reg)
    text = "\n".join(h.lines)
    assert "region" in text and "cpu [s]" in text and "cpu/wall" in text
    assert "step" in text
    assert "cost is cpu [s]" in text                    # the reading instruction


def test_summary_of_an_empty_registry_is_silent(cap):
    log, h = cap
    summary(log, registry=TimingRegistry())
    assert h.lines == []


def test_global_registry_reset():
    with timer("scratch"):
        pass
    assert REGISTRY.get("scratch") is not None
    reset()
    assert REGISTRY.get("scratch") is None
