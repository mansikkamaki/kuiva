"""Shared pytest fixtures/markers for the Kuiva test suite."""
import logging
import sys
import warnings
from pathlib import Path

import pytest

#: BLAS threads the suite and the reference generators are expected to run at (part of the reference contract). Applied by ``external/env.sh``; only *checked* here — see ``_report_thread_width``.
THREAD_CAP = 4

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import stages                                                             # noqa: E402

DIRAC_BASIS = REPO / "external/install/dirac/share/dirac/basis"
MOLCAS_ANORCC = REPO / "external/install/openmolcas/basis_library/ANO-RCC"


def external_basis_available() -> bool:
    """True if the DIRAC/OpenMolcas basis libraries are present (Tier-2 generation)."""
    return DIRAC_BASIS.is_dir() and MOLCAS_ANORCC.is_file()


@pytest.fixture
def kuiva_caplog(caplog):
    """``caplog``, but able to see Kuiva's records.

    ``kuiva.util.logging`` sets ``propagate = False`` on the ``kuiva`` logger on purpose: the
    INFO stream is the program's output and must not be duplicated through whatever
    handlers an embedding application has on the root logger. pytest's ``caplog`` captures at
    the root, so it sees nothing without this. Attaching its handler directly to ``kuiva`` is
    the fix that does not compromise the policy being tested.

    Use it wherever a test asserts on a WARNING/ERROR — and several do, because "proceeds but
    the user should know" is a behaviour, not a decoration.
    """
    logger = logging.getLogger("kuiva")
    level = logger.level
    logger.addHandler(caplog.handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield caplog
    finally:
        logger.removeHandler(caplog.handler)
        logger.setLevel(level)


@pytest.fixture(autouse=True)
def _fresh_memory_budget():
    """Clear the process-global memory budget between tests.

    :data:`kuiva.util.resources.BUDGET` is process-global by design — it has to be, since the
    arrays it accounts for are. Without this fixture reservations accumulate across tests and
    a later test fails on a limit an earlier one filled, which would be a maddening thing to
    debug. The *limits* are left in place: they come from the committed ``defaults.conf`` and
    are what the suite is meant to run under.
    """
    from kuiva.util import resources

    resources.BUDGET.clear()
    yield
    resources.BUDGET.clear()


@pytest.fixture(scope="session", autouse=True)
def _report_thread_width():
    """⚠ Report the BLAS thread width, and warn if it exceeds the reference cap of 4.

    This **checks**, it does not set. Setting ``OMP_NUM_THREADS`` from here would be a lie:
    MKL reads it when it is loaded, which has already happened by the time any fixture runs,
    so an assignment here would change the environment and not the behaviour — the worst kind
    of guard, one that reports success while doing nothing. ``external/env.sh`` is where the
    cap is actually applied (the one script sourced before any run).

    It matters for reference data specifically: committed records in ``tests/reference/``
    carry ``cpu_seconds``/``wall_seconds``, and a record generated at one width and compared
    against a run at another is not a comparison. A warning here is what makes a mismatched
    width visible instead of silently baked into a regenerated file.

    ⚠ **Queried through the OpenMP runtime, NOT through MKL, and that distinction is the
    whole reason this fixture works.** ``MKL_Get_Max_Threads`` self-caps at the number of
    *physical* cores — measured here: requests of 6, 8 and 16 all report 4 — so a guard
    written against it can never fire on this machine and would be a check that cannot fail
    (the same accounting rule, in a different place). ``omp_get_max_threads`` tracks the request
    faithfully (2 -> 2, 8 -> 8, 16 -> 16) and reads **8** when the variable is unset, which
    is the state this cap exists to catch.

    That difference is also what the cap actually buys. MKL was already computing on 4
    threads; the other four were OpenMP threads **spin-waiting**, which `process_time` counts
    in full. Capping removes them: CPU seconds roughly halve while wall time does
    not move.
    """
    import ctypes
    import os

    # Both names are reported: KUIVA_NUM_THREADS is the knob, but what
    # the OpenMP runtime below actually obeys is OMP_NUM_THREADS, and a run where the two
    # disagree is exactly the state worth naming in the warning.
    requested = "KUIVA_NUM_THREADS={} OMP_NUM_THREADS={}".format(
        os.environ.get("KUIVA_NUM_THREADS", "(unset)"),
        os.environ.get("OMP_NUM_THREADS", "(unset)"))
    granted = None
    for soname in ("libiomp5.so", "libgomp.so.1", "libomp.so"):
        try:
            granted = ctypes.CDLL(soname).omp_get_max_threads()
            break
        except OSError:                                          # no OpenMP runtime; fine
            continue

    if granted is not None and granted > THREAD_CAP:
        warnings.warn(
            "BLAS is running {} threads ({}), above the reference thread cap of {}. "
            "Source external/env.sh, or export KUIVA_NUM_THREADS={} (and OMP_NUM_THREADS "
            "with it) — reference data regenerated at this width will not match records "
            "made at the cap."
            .format(granted, requested, THREAD_CAP, THREAD_CAP), RuntimeWarning)
    yield


@pytest.fixture(scope="session", autouse=True)
def _no_persistent_amf_cache():
    """⚠ **The suite never touches the user's X2CAMF cache**.

    Two independent reasons, and the second is the one that makes this mandatory rather than
    tidy:

    * **Pollution.** Running ``pytest`` would otherwise leave a directory of corrections
      behind, including ones built at a modified speed of light or from a deliberately broken
      solution.
    * ⚠ **Result dependence.** Several tests assert "one four-component solve per unique
      element" by **call count** — the only honest way to check caching (timing is not
      evidence). With a persistent cache in play, ``atomic.clear_cache()`` no longer produces a
      cold start, so those counts would read zero on a developer's machine and N on a fresh
      clone. A test whose outcome depends on what is already on the disk is not a test.

    The **in-process** cache is untouched, so a session still solves each element once and the
    suite costs what it always did. ``tests/test_amf_cache.py`` overrides this per test with
    its own temporary directory, which is the only place the persistent cache is exercised.

    ⚠ **This sets the session default; ``_amf_stage_cache`` below overrides it per test** when
    stage checkpoints are enabled. Both reasons above survive that override — the
    directory is inside ``tests/checkpoints`` rather than the user's, and it is switched off
    for exactly the tests whose subject the cache would hide.

    ⚠ **Session scope, and it has to be.** As a function-scoped fixture this silently did not
    work: pytest sets higher-scoped fixtures up first, so a *module*-scoped fixture that builds
    a correction (``test_amf_molecular.hf_molecule``) ran before the environment variable was
    set and read the developer's real cache — which presents as a call-count assertion failing
    with ``0 == 1`` on one machine and passing on another. Exactly the result dependence this
    fixture exists to prevent, produced by the fixture itself.
    """
    from kuiva.amf import cache

    patch = pytest.MonkeyPatch()
    patch.setenv(cache.ENV_CACHE_DIR, "off")
    yield
    patch.undo()


requires_external = pytest.mark.skipif(
    not external_basis_available(),
    reason="DIRAC/OpenMolcas basis libraries not installed (Tier-2 only)",
)


# --- stage checkpoints (tests/stages.py) ---------------------------------

def pytest_addoption(parser):
    parser.addoption("--checkpoints", action="store", default=None, choices=stages.MODES,
                     help="replay disk-stored intermediate stages of the heavy calculations: "
                          "off (default) | on | read | refresh | trust. See tests/stages.py "
                          "and tests/stages.py.")
    parser.addoption("--checkpoint-dir", action="store", default=None,
                     help="where stage checkpoints live (default tests/checkpoints)")


def pytest_configure(config):
    if config.getoption("--checkpoint-dir"):
        stages.set_directory(config.getoption("--checkpoint-dir"))
    if config.getoption("--checkpoints"):
        stages.set_mode(config.getoption("--checkpoints"))


@pytest.fixture(autouse=True)
def _stage_under_test(request):
    """⚠ **A test is never served a checkpoint of the stage it is testing**.

    A checkpoint replaces a computation, so replaying one into the test whose assertions are
    *about* that computation would make the test assert on numbers a previous run produced —
    it would pass whatever the code now does. ``@pytest.mark.stage_under_test("amf_atomic")``
    declares the subject; that stage and everything downstream of it are then computed for
    real, however fresh the stored entries are.

    The source fingerprint (:func:`stages.stage_fingerprint`) is the *other*, automatic gate:
    it invalidates a checkpoint when the code that produced it changed. This one covers the
    case where nothing changed and the test is still the thing that has to run.
    """
    names = []
    for mark in request.node.iter_markers("stage_under_test"):
        names.extend(mark.args)
    previous = stages.set_under_test(names)
    try:
        yield
    finally:
        stages.restore_under_test(previous)


@pytest.fixture(autouse=True)
def _amf_stage_cache(_stage_under_test, monkeypatch):
    """Point Kuiva's own X2CAMF correction cache at a fingerprinted checkpoint directory.

    This is what actually removes the four-component atomic solve from a test that only
    *consumes* a correction, because that solve happens deep inside ``amf_correction`` and no
    test-level checkpoint can reach it. :func:`stages.amf_cache_directory` decides whether it
    is safe for this test and returns ``None`` when it is not; ``off`` (the default) leaves
    the session policy of :func:`_no_persistent_amf_cache` in place unchanged.

    ⚠ Depends on ``_stage_under_test`` so the marks are in place before the directory is
    chosen — the same fixture-ordering trap that made ``_no_persistent_amf_cache``
    session-scoped.

    ⚠ **A ``refresh`` pass can leave a gap, by construction.** An in-process hit short-circuits
    the disk write (``kuiva.amf.atomic.atomic_correction``), so an element whose correction is
    first computed inside a *subject* test — where this cache is deliberately off — is never
    written. Clearing the in-process cache per test would close it and is the wrong trade: the
    solutions cache has no disk counterpart, so tests calling ``atomic_solution`` directly
    (``test_amf_basis``, ``test_amf_decouple``) would re-solve on every one. The supported fix
    is a second pass in ``--checkpoints=on``, which does not purge and does write.
    """
    from kuiva.amf import cache

    directory = stages.amf_cache_directory()
    if directory is None:
        return
    if stages.mode() == "refresh" and stages.once("amf-cache-purge"):
        cache.purge(directory)
    directory.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv(cache.ENV_CACHE_DIR, str(directory))


def pytest_terminal_summary(terminalreporter):
    """Report what was replayed and what was computed — never silently."""
    lines = stages.summary_lines()
    if lines:
        terminalreporter.write_sep("-", "stage checkpoints")
        for line in lines:
            terminalreporter.write_line(line)
