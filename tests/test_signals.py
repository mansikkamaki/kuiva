"""Stopping on a signal: the kill that was never announced.

What each group can fail on:

* ⚠ **the handler does nothing but set a flag** — asserted by capturing the log while it
  runs. It is called between bytecodes in the main thread, so it can interrupt code holding
  the logging lock, and logging from there deadlocks the run it exists to save;
* **installation is a loan, not a seizure** — the previous disposition comes back when the
  stage ends, including when it ends by raising, and a handler that cannot be installed at
  all (off the main thread) is refused rather than silently skipped;
* ⚠ **a second signal is not waited for.** The escalation is driven through the one seam in
  the module, because a test cannot let the real one run: it kills the interpreter;
* ⚠ **the request outlives the stage that caught it**, which is what makes a signalled run
  *exit* rather than pause — the next long stage refuses instead of starting work the
  process will not live to finish;
* **the stop and the write are one decision**, exactly as the deadline's are: the cadence
  in the end-to-end tests suppresses every ordinary write, so a file existing at all proves
  the signal forced it.

⚠ Every test here sends real signals to the test process. ``tests/conftest.py`` clears the
process-global request around every test in the suite; without that, one test here would
make a dozen unrelated ones several files later refuse to start.
"""
import itertools
import os
import signal
import threading

import numpy as np
import pytest

from kuiva.ci.strings import Determinants, hamiltonian_matrix, rdm12
from kuiva.integrals.transform import ThreeIndexAO
from kuiva.io.checkpoint import CheckpointPolicy, read_checkpoint
from kuiva.mcscf.orbopt import CASIntegrals, OrbitalSpaces, cas_energy, optimize_orbitals
from kuiva.util.deadline import Deadline
from kuiva.util.signals import (DEFAULT_SIGNALS, SignalStop, StopRequested, clear, name_of,
                                pending, raise_if_pending, resolve_signal_set)

#: ⚠ USR1 throughout, never TERM: a test that sends TERM and finds the handler was not
#: installed kills the pytest process instead of failing.
TEST_SIGNAL = signal.SIGUSR1


# --- what signals= accepts ----------------------------------------------------------------

def test_the_signal_set_is_read_from_names_or_numbers():
    assert resolve_signal_set(True) == tuple(getattr(signal, "SIG" + n) for n in
                                             DEFAULT_SIGNALS)
    assert resolve_signal_set("TERM") == (signal.SIGTERM,)
    assert resolve_signal_set("SIGTERM") == (signal.SIGTERM,)
    assert resolve_signal_set(("usr1", signal.SIGUSR2)) == (signal.SIGUSR1, signal.SIGUSR2)
    assert resolve_signal_set((signal.SIGTERM, "TERM")) == (signal.SIGTERM,)   # deduplicated


def test_sigint_is_not_in_the_default_set():
    """⚠ Ctrl-C is expected to interrupt *now*. Deferring it to the end of a macro-iteration
    is exactly the surprise an opt-in handler exists to avoid, so it is opt-in twice over."""
    assert "INT" not in DEFAULT_SIGNALS
    assert signal.SIGINT not in resolve_signal_set(True)
    assert resolve_signal_set("INT") == (signal.SIGINT,)          # ... but nameable


@pytest.mark.parametrize("bad", ["NOSUCH", "SIGNOSUCH", 3.5, {}])
def test_an_unknown_signal_is_refused(bad):
    with pytest.raises((ValueError, TypeError)):
        resolve_signal_set(bad)


def test_nothing_is_installed_unless_it_is_asked_for():
    """⚠ The user decision this module is built on: a library that installs signal handlers
    behind your back breaks embedding, test runners and notebooks."""
    assert SignalStop.resolve(None) is None
    assert SignalStop.resolve(False) is None
    stopper = SignalStop.resolve(True)
    assert stopper is not None and not stopper.installed
    assert SignalStop.resolve(stopper) is stopper


# --- installation is a loan ----------------------------------------------------------------

def test_the_previous_handler_comes_back():
    def mine(signum, frame):                          # pragma: no cover - never called
        raise AssertionError("the previous handler must not run while Kuiva's is installed")

    previous = signal.signal(TEST_SIGNAL, mine)
    try:
        with SignalStop(("USR1",)) as stopper:
            assert stopper.installed
            assert signal.getsignal(TEST_SIGNAL) is not mine
        assert signal.getsignal(TEST_SIGNAL) is mine
    finally:
        signal.signal(TEST_SIGNAL, previous)


def test_the_previous_handler_comes_back_when_the_run_raises():
    """The restore is in ``__exit__``, so an exception in the middle of a stage cannot leave
    a handler behind pointing into a calculation that no longer exists."""
    before = signal.getsignal(TEST_SIGNAL)
    with pytest.raises(ZeroDivisionError):
        with SignalStop(("USR1",)):
            raise ZeroDivisionError
    assert signal.getsignal(TEST_SIGNAL) is before


def test_installing_off_the_main_thread_is_refused():
    """⚠ Refused rather than warned about: Python cannot install a handler there at all, and
    a caller who believes the run is protected when it is not is the failure this prevents."""
    failures = []

    def build():
        try:
            SignalStop(("USR1",))
        except RuntimeError as exc:
            failures.append(str(exc))

    thread = threading.Thread(target=build)
    thread.start()
    thread.join()
    assert failures and "main thread" in failures[0]


# --- the handler itself ---------------------------------------------------------------------

def test_the_handler_sets_a_flag_and_logs_nothing(kuiva_caplog):
    """⚠ The rule that keeps the handler safe. It runs between bytecodes in the main thread,
    so it may be interrupting code that holds the logging lock; logging from inside it
    deadlocks the process it was meant to save. Everything the user sees is printed later,
    from ordinary code at the macro-iteration boundary."""
    with SignalStop(("USR1",)) as stopper:
        assert stopper.requested is None
        before = len(kuiva_caplog.records)             # installation itself says so at DEBUG
        os.kill(os.getpid(), TEST_SIGNAL)
        assert len(kuiva_caplog.records) == before     # ... the handler says nothing at all
        assert stopper.requested is not None
        assert stopper.requested.signum == TEST_SIGNAL
        assert stopper.should_stop() is True


def test_the_request_outlives_the_stage_that_caught_it():
    """⚠ Process-level on purpose: a signal is delivered to the process, not to a stage, and
    this is what makes a signalled run exit rather than merely pause."""
    with SignalStop(("USR1",)):
        os.kill(os.getpid(), TEST_SIGNAL)
    assert pending() is not None
    with pytest.raises(StopRequested, match="was not started"):
        raise_if_pending("this NEVPT2")
    clear()
    assert pending() is None
    raise_if_pending("this NEVPT2")                    # and now it starts


def test_a_second_signal_is_not_waited_for():
    """⚠ The first signal is a request, the second an order. Driven through the module's one
    seam, because the real escalation re-raises and would kill the test process — which is
    the point of it."""
    escalations = []
    stopper = SignalStop(("USR1",))
    stopper._escalate = lambda signum: escalations.append(signum)
    with stopper:
        os.kill(os.getpid(), TEST_SIGNAL)
        assert not escalations
        os.kill(os.getpid(), TEST_SIGNAL)
    assert escalations == [TEST_SIGNAL]
    assert pending().count == 2


def test_a_signal_stop_with_no_signals_is_refused():
    with pytest.raises(ValueError, match="catches nothing"):
        SignalStop(())


def test_names_are_printable():
    assert name_of(signal.SIGTERM) == "SIGTERM"


# --- what it does to a real optimization ---------------------------------------------------

@pytest.fixture(scope="module")
def system():
    """A small synthetic two-component problem with a frozen, deterministic RDM solver."""
    rng = np.random.default_rng(3)
    nao, n_elec = 6, 2
    n = 2 * nao
    factors = ThreeIndexAO(l_packed=rng.standard_normal((3 * nao, nao * (nao + 1) // 2)),
                           nao=nao, origin="cholesky")
    h_ao = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    h_ao = 0.5 * (h_ao + h_ao.conj().T)
    _, c0 = np.linalg.eigh(h_ao)
    spaces = OrbitalSpaces.from_counts(n_inactive=2, n_active=4, n_orb=n)
    c0 = np.ascontiguousarray(c0)

    ints = CASIntegrals.build(factors, h_ao, c0, spaces)
    dets = Determinants.from_occupations(
        itertools.combinations(range(spaces.n_active), n_elec), spaces.n_active)
    matrix = hamiltonian_matrix(dets, ints.h_active_effective(), ints.active_eri())
    _, vectors = np.linalg.eigh(matrix.toarray())
    gamma, gamma2 = rdm12(dets, vectors[:, :1])

    def solve(ints_at):
        return cas_energy(ints_at, gamma, gamma2), gamma, gamma2

    return factors, h_ao, c0, spaces, solve


def signalling_solver(solve, at_call=1):
    """``solve``, sending the test signal from inside the ``at_call``-th CI solve.

    Which is where a real one arrives: in the middle of an iteration, not at its boundary.
    ⚠ Counted in *solves*, not macro-iterations — an iteration takes as many as its trial
    steps need — so the tests using it signal on the first one, where the two coincide.
    """
    state = {"calls": 0}

    def wrapped(ints):
        state["calls"] += 1
        if state["calls"] == at_call:
            os.kill(os.getpid(), TEST_SIGNAL)
        return solve(ints)

    return wrapped


def signalling_callback(at_iteration):
    """Send the test signal from the optimizer's own hook, after ``at_iteration`` completes.

    The policy has already asked by then, so this one lands at the *following* boundary —
    which is the more interesting shape: it is what an unannounced kill mid-run looks like.
    """
    def callback(info):
        if info["iteration"] == at_iteration:
            os.kill(os.getpid(), TEST_SIGNAL)
        return None

    return callback


def test_a_signal_stops_the_run_and_forces_the_final_checkpoint(system, tmp_path,
                                                                kuiva_caplog):
    """⚠ The whole claim. The cadence would suppress every write (an hour of minimum
    interval, a cost fraction of nothing), so the file existing proves the signal forced it
    — and it holds the iteration the signal arrived in, completed, not abandoned."""
    factors, h_ao, c0, spaces, solve = system
    path = tmp_path / "run.chk"
    with SignalStop(("USR1",)) as stopper:
        policy = CheckpointPolicy(path, min_interval=3600.0, cost_fraction=1e-12,
                                  n_active_elec=2, signals=stopper)
        result = optimize_orbitals(factors, h_ao, c0, spaces,
                                   signalling_solver(solve, at_call=1), e_nuc=0.0,
                                   max_iter=20, conv_grad=1e-10, conv_energy=1e-12,
                                   report=False, callback=policy.callback)

    assert result.n_iterations == 1 and not result.converged
    assert stopper.fired
    assert path.exists() and read_checkpoint(path).iteration == 1
    assert any("SIGUSR1" in r.getMessage() for r in kuiva_caplog.records)


def test_the_run_resumes_from_what_the_signal_left(system, tmp_path):
    """A signalled run keeps everything it finished, which is the difference between this
    and being killed mid-iteration."""
    factors, h_ao, c0, spaces, solve = system
    kwargs = dict(e_nuc=0.0, conv_grad=1e-10, conv_energy=1e-12, report=False)
    straight = optimize_orbitals(factors, h_ao, c0, spaces, solve, max_iter=12, **kwargs)

    path = tmp_path / "run.chk"
    with SignalStop(("USR1",)) as stopper:
        # signalled from the hook after iteration 2, so it is acted on at the next boundary
        policy = CheckpointPolicy(path, min_interval=3600.0, cost_fraction=1e-12,
                                  n_active_elec=2, signals=stopper,
                                  chain=signalling_callback(at_iteration=2))
        stopped = optimize_orbitals(factors, h_ao, c0, spaces, solve, max_iter=12,
                                    callback=policy.callback, **kwargs)
    assert stopped.n_iterations == 3
    checkpoint = read_checkpoint(path)
    assert checkpoint.iteration == 3

    clear()                                            # the next job is a new process
    resumed = optimize_orbitals(
        factors, h_ao, checkpoint.coeff, checkpoint.spaces, solve, max_iter=12,
        **dict(kwargs, **checkpoint.optimizer_kwargs(space_key=None)))
    assert resumed.n_iterations == straight.n_iterations
    assert resumed.energy == pytest.approx(straight.energy, abs=1e-10)


def test_a_signal_names_itself_rather_than_the_deadline(system, tmp_path, kuiva_caplog):
    """Both stop causes on one hook. The signal is asked first — it is already on its way,
    so there is no arithmetic to do — and the warning has to say which one it was, because
    "ran out of time" and "was cancelled" call for different things next."""
    factors, h_ao, c0, spaces, solve = system
    with SignalStop(("USR1",)) as stopper:
        policy = CheckpointPolicy(tmp_path / "run.chk", min_interval=0.0, cost_fraction=1.0,
                                  n_active_elec=2, signals=stopper,
                                  deadline=Deadline.after("10h"))
        result = optimize_orbitals(factors, h_ao, c0, spaces,
                                   signalling_solver(solve, at_call=1), e_nuc=0.0,
                                   max_iter=20, conv_grad=1e-10, conv_energy=1e-12,
                                   report=False, callback=policy.callback)
    assert result.n_iterations == 1
    messages = [r.getMessage() for r in kuiva_caplog.records]
    assert any("SIGUSR1" in m for m in messages)
    assert not any("deadline" in m for m in messages)


def test_without_a_checkpoint_it_still_stops_and_says_nothing_was_saved(system,
                                                                       kuiva_caplog):
    factors, h_ao, c0, spaces, solve = system
    with SignalStop(("USR1",)) as stopper:
        result = optimize_orbitals(factors, h_ao, c0, spaces,
                                   signalling_solver(solve, at_call=1), e_nuc=0.0,
                                   max_iter=20, conv_grad=1e-10, conv_energy=1e-12,
                                   report=False, callback=stopper.as_callback())
    assert result.n_iterations == 1 and not result.converged
    assert any("NO checkpoint" in r.getMessage() for r in kuiva_caplog.records)


def test_a_converged_iteration_is_not_a_signal_stop(system, tmp_path):
    """⚠ A run that finished is not a run that was cancelled, even if the signal landed on
    the last iteration: "stopped by a signal" is what tells a reader the result is an
    iterate rather than an answer."""
    factors, h_ao, c0, spaces, solve = system
    grabbed = {}

    def grab(info):
        grabbed.update(info)
        return None

    optimize_orbitals(factors, h_ao, c0, spaces, solve, e_nuc=0.0, max_iter=1,
                      conv_grad=1e-10, conv_energy=1e-12, report=False, callback=grab)

    with SignalStop(("USR1",)) as stopper:
        os.kill(os.getpid(), TEST_SIGNAL)
        policy = CheckpointPolicy(tmp_path / "converged.chk", min_interval=3600.0,
                                  cost_fraction=1e-12, n_active_elec=2, signals=stopper)
        assert policy.callback(dict(grabbed, converged=True)) is None
        assert not stopper.fired
        assert policy.callback(dict(grabbed, converged=False)) is False
        assert stopper.fired
    # the converged iteration was written anyway, by its own rule
    assert (tmp_path / "converged.chk").exists()
