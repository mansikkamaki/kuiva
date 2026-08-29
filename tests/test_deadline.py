"""The wall-clock deadline: reading a queue's limit, and stopping in time to save.

What each group can fail on:

* **the duration parser** — including the one refusal that is the point of it: a bare
  numeric string is sixty *minutes* to Slurm's ``--time`` and sixty *seconds* here, and a
  factor of sixty in a deadline ends an allocation with nothing written;
* **the queue probes** — the Slurm environment variable, the ``scontrol`` fallback (driven
  through a fake binary on ``PATH``, so the argv and the parsing are both exercised), and
  the two spellings of "no limit". ⚠ A probe that cannot answer must never raise; what
  refuses is the *explicit request* one level up;
* ⚠ **the arithmetic that decides**, which is predictive and not reactive: stopping when the
  time is already spent is stopping too late, because the checkpoint is written afterwards.
  The scripted-clock tests assert the decision rather than the machine's speed;
* ⚠ **that the stop and the write are one decision** — the final checkpoint lands *past* a
  cadence that would otherwise have suppressed it (``min_interval`` an hour,
  ``cost_fraction`` effectively zero: nothing but a forced write can land), and the run
  resumes from it. That is the whole claim of the feature.

Tolerances: none of this is numerical. The one time-based assertion (a budget's remaining
time) is a generous band, because a test process can be descheduled for a second.
"""
import itertools
import os
import time

import numpy as np
import pytest

from kuiva.ci.strings import Determinants, hamiltonian_matrix, rdm12
from kuiva.integrals.transform import ThreeIndexAO
from kuiva.io.checkpoint import CheckpointPolicy, read_checkpoint
from kuiva.mcscf.orbopt import CASIntegrals, OrbitalSpaces, cas_energy, optimize_orbitals
from kuiva.util.deadline import (DEFAULT_SAFETY_SECONDS, Deadline, QueueLimit, detect_queue,
                                 format_duration, parse_walltime, queue_limit)

SLURM_ENV = ("SLURM_JOB_ID", "SLURM_JOBID", "SLURM_JOB_END_TIME")


@pytest.fixture(autouse=True)
def _no_inherited_job(monkeypatch):
    """Every test starts outside a batch job, whatever the machine running the suite is.

    ⚠ Not a convenience: the suite itself may be running inside a Slurm allocation, and a
    test that silently read *that* job's limit would assert something different there.
    """
    for name in SLURM_ENV:
        monkeypatch.delenv(name, raising=False)


# --- the duration parser ----------------------------------------------------------------

@pytest.mark.parametrize("value, seconds", [
    (3600, 3600.0), (90.5, 90.5),
    ("45s", 45.0), ("90m", 5400.0), ("6h", 21600.0), ("2d", 172800.0),
    ("1h30m", 5400.0), ("1h 30m", 5400.0), ("1d12h", 129600.0),
    ("24:00:00", 86400.0), ("2-12:00:00", 216000.0), ("90:00", 5400.0),
    ("0:30", 30.0),
])
def test_a_wall_time_is_read_from_the_forms_a_batch_script_already_contains(value, seconds):
    assert parse_walltime(value) == pytest.approx(seconds)


def test_a_bare_numeric_string_is_refused_because_slurm_reads_it_as_minutes():
    """⚠ The refusal this parser exists for. `--time=60` is an hour to Slurm and would be a
    minute here; the two readings differ by the factor that decides whether anything gets
    written. Both fixes are named in the message."""
    with pytest.raises(ValueError, match="sixty minutes"):
        parse_walltime("60")
    # the same number *as a number* is unambiguous and accepted
    assert parse_walltime(60) == 60.0


@pytest.mark.parametrize("bad", ["", "soon", "-5", "1h30", 0, -10, True])
def test_an_unreadable_wall_time_is_refused(bad):
    with pytest.raises((ValueError, TypeError)):
        parse_walltime(bad)


def test_durations_are_printed_in_ascii_and_coarsely():
    assert format_duration(20520) == "5h 42m"
    assert format_duration(95) == "1m 35s"
    assert format_duration(float("inf")) == "unlimited"
    assert format_duration(180000).startswith("2d")
    assert all(ord(c) < 128 for c in format_duration(20520))


# --- no deadline is the default ----------------------------------------------------------

def test_there_is_no_default_deadline():
    """⚠ The load-bearing default. A cluster with no queue limit is an ordinary place to
    run, and an invented deadline could only end such a run early for no reason."""
    assert Deadline.resolve(None) is None
    assert Deadline.resolve("none") is None


def test_auto_outside_a_queue_reports_no_limit_instead_of_refusing():
    """``"auto"`` is the portable spelling: it must work unchanged on a laptop."""
    deadline = Deadline.resolve("auto")
    assert deadline is not None and deadline.unlimited
    assert deadline.remaining() == float("inf")
    assert deadline.should_stop(write_seconds=1e6) is False
    assert deadline.ends_at() is None
    assert not deadline.too_late_to_start()


def test_a_named_queue_that_cannot_be_read_refuses(monkeypatch):
    """⚠ The opposite decision, and for the opposite reason: an explicit request that
    quietly produced no deadline is a job killed at the wall twelve hours later with
    nothing in the output ever having said the request failed."""
    with pytest.raises(ValueError, match="could not be read"):
        Deadline.resolve("slurm")
    with pytest.raises(ValueError, match="could not be read"):
        Deadline.resolve("queue")
    # and the message names both ways out
    try:
        Deadline.resolve("slurm")
    except ValueError as exc:
        assert "deadline='6h'" in str(exc) and "deadline='auto'" in str(exc)


def test_a_deadline_object_and_a_budget_both_resolve():
    explicit = Deadline.after("2h")
    assert Deadline.resolve(explicit) is explicit
    assert Deadline.resolve("2h").remaining() == pytest.approx(7200.0, abs=5.0)
    assert Deadline.resolve(1800).remaining() == pytest.approx(1800.0, abs=5.0)


# --- reading the limit out of Slurm -------------------------------------------------------

def test_the_slurm_limit_comes_from_the_environment_first(monkeypatch):
    """⚠ Preferred over ``scontrol`` deliberately: it is free, needs no client binaries,
    contacts no controller, and is an unambiguous UNIX timestamp rather than a local time
    string that is an hour ambiguous across a DST fold."""
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    monkeypatch.setenv("SLURM_JOB_END_TIME", str(int(time.time()) + 3600))
    assert detect_queue() == "slurm"

    limit = queue_limit("slurm")
    assert limit.system == "Slurm" and limit.job == "4242"
    assert limit.source == "SLURM_JOB_END_TIME"
    assert not limit.unlimited

    deadline = Deadline.resolve("slurm")
    assert deadline.remaining() == pytest.approx(3600.0, abs=5.0)
    assert "4242" in deadline.source


def test_an_end_time_decades_away_is_no_limit_at_all(monkeypatch):
    """⚠ An unlimited Slurm job carries a sentinel far in the future rather than nothing;
    without this it would be carried around as a deadline in 2106."""
    monkeypatch.setenv("SLURM_JOB_ID", "7")
    monkeypatch.setenv("SLURM_JOB_END_TIME", str(int(time.time()) + 40 * 365 * 86400))
    assert queue_limit("slurm").unlimited
    deadline = Deadline.resolve("slurm")
    assert deadline.unlimited and deadline.should_stop() is False


def _fake_scontrol(tmp_path, monkeypatch, body):
    """Put a fake ``scontrol`` first on ``PATH``, so argv and parsing are both exercised."""
    script = tmp_path / "scontrol"
    script.write_text("#!/bin/sh\n" + body + "\n")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))


def test_the_limit_falls_back_to_scontrol(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "99")
    end = time.time() + 7200.0
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(end))
    _fake_scontrol(tmp_path, monkeypatch,
                   'echo "JobId=99 JobName=kuiva EndTime={} Partition=test"'.format(stamp))
    limit = queue_limit("slurm")
    assert limit.source == "scontrol EndTime"
    assert limit.end_time == pytest.approx(end, abs=1.5)
    assert Deadline.resolve("slurm").remaining() == pytest.approx(7200.0, abs=5.0)


def test_scontrol_saying_unknown_is_a_job_with_no_time_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "99")
    _fake_scontrol(tmp_path, monkeypatch,
                   'echo "JobId=99 TimeLimit=UNLIMITED EndTime=Unknown"')
    assert queue_limit("slurm").unlimited
    assert Deadline.resolve("slurm").unlimited


def test_a_probe_that_fails_is_a_miss_and_never_an_exception(tmp_path, monkeypatch):
    """⚠ A failure to *measure* must not fail a calculation. The probe returns nothing; the
    refusal belongs to the explicit request above, and ``"auto"`` carries on."""
    monkeypatch.setenv("SLURM_JOB_ID", "99")
    _fake_scontrol(tmp_path, monkeypatch, 'echo "slurm_load_jobs error" >&2; exit 1')
    assert queue_limit("slurm") is None
    assert Deadline.resolve("auto").unlimited
    with pytest.raises(ValueError):
        Deadline.resolve("slurm")


def test_an_unknown_queue_system_is_refused_by_name():
    with pytest.raises(ValueError, match="unknown queue system"):
        queue_limit("pbs")


def test_the_registry_is_the_extension_point(monkeypatch):
    """A second queue system is two functions and an entry — nothing else in the program
    learns a scheduler's name."""
    from kuiva.util import deadline as mod

    end = time.time() + 600.0
    monkeypatch.setitem(mod.QUEUE_SYSTEMS, "pbs",
                        (lambda: "1.pbs",
                         lambda job: QueueLimit("PBS", job, end_time=end, source="qstat")))
    assert queue_limit("pbs").end_time == end
    assert Deadline.resolve("pbs").remaining() == pytest.approx(600.0, abs=5.0)


# --- the arithmetic that decides ---------------------------------------------------------

def test_the_prediction_is_the_longest_recent_iteration_not_the_mean():
    """⚠ Iterations are not uniform — a second-order escalation is several times a
    quasi-Newton step — and under-predicting is the one error that gets the job killed."""
    deadline = Deadline.after("1h", safety=0.0, window=3)
    assert deadline.predicted() == 0.0
    for seconds in (10.0, 90.0, 20.0):
        deadline.observe(seconds)
    assert deadline.predicted() == 90.0
    deadline.observe(30.0)                     # the window slides, oldest first
    assert deadline.predicted() == 90.0
    deadline.observe(5.0)                      # ... and now the 90 s iteration has left it
    assert deadline.predicted() == 30.0


def test_the_stop_rule_is_predictive_rather_than_reactive():
    """⚠ The rule the module exists for. With 100 s left, a 40 s iteration and a 10 s write,
    stopping is right — the next iteration would end 10 s past the wall with its checkpoint
    unwritten — even though none of the budget is spent yet."""
    deadline = Deadline(epoch=time.time() + 100.0, safety=30.0)
    deadline.observe(40.0)
    assert deadline.reserve(write_seconds=10.0) == pytest.approx(80.0)
    assert deadline.should_stop(write_seconds=10.0) is False       # 100 > 80, one more fits
    assert deadline.should_stop(write_seconds=40.0) is True        # 100 < 110, it does not
    later = Deadline(epoch=time.time() + 60.0, safety=30.0)
    later.observe(40.0)
    assert later.should_stop() is True                             # 60 < 70, and nothing spent


def test_starting_at_all_is_refused_when_only_the_margin_is_left():
    deadline = Deadline(epoch=time.time() + 10.0, safety=60.0)
    assert deadline.too_late_to_start()
    with pytest.raises(RuntimeError, match="was not started"):
        deadline.assert_room("this CASSCF")
    assert not Deadline(epoch=time.time() + 3600.0, safety=60.0).too_late_to_start()


def test_the_safety_margin_is_stated_and_not_hidden():
    assert DEFAULT_SAFETY_SECONDS > 0.0
    assert Deadline.after("1h").safety == DEFAULT_SAFETY_SECONDS


def test_a_deadline_is_an_instant_or_a_budget_and_not_both():
    with pytest.raises(ValueError, match="not both and not neither"):
        Deadline(epoch=time.time(), duration=10.0)
    with pytest.raises(ValueError, match="not both and not neither"):
        Deadline()


# --- what it does to a real optimization -------------------------------------------------

class Countdown(Deadline):
    """A deadline on a scripted clock: it expires after ``stop_after`` macro-iterations.

    ⚠ The end-to-end tests assert the *decision*, never the machine's speed. A test tied to
    real iteration times would pass or fail with the load on the box.
    """

    def __init__(self, stop_after, **kwargs):
        super().__init__(duration=1e9, source="a scripted clock", safety=0.0, **kwargs)
        self.stop_after = int(stop_after)
        self.seen = 0

    def observe(self, seconds):
        super().observe(seconds)
        self.seen += 1

    def remaining(self):
        return 0.0 if self.seen >= self.stop_after else 1e9


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


def test_the_deadline_stops_the_run_and_forces_the_final_checkpoint(system, tmp_path,
                                                                    kuiva_caplog):
    """⚠ The whole claim, in one test. The cadence here would suppress every write
    (``min_interval`` an hour, ``cost_fraction`` effectively zero), so a file existing at all
    proves the deadline forced it — and it is the *last completed* iteration, which is what
    makes the stop worth making."""
    factors, h_ao, c0, spaces, solve = system
    path = tmp_path / "run.chk"
    deadline = Countdown(stop_after=2)
    policy = CheckpointPolicy(path, min_interval=3600.0, cost_fraction=1e-12,
                              n_active_elec=2, deadline=deadline)
    result = optimize_orbitals(factors, h_ao, c0, spaces, solve, e_nuc=0.0, max_iter=20,
                               conv_grad=1e-10, conv_energy=1e-12, report=False,
                               callback=policy.callback)

    assert result.n_iterations == 2 and not result.converged
    assert deadline.fired
    assert path.exists() and read_checkpoint(path).iteration == 2
    assert any("deadline" in r.getMessage() for r in kuiva_caplog.records)
    assert any("scripted clock" in r.getMessage() for r in kuiva_caplog.records)


def test_a_run_stopped_by_the_deadline_restarts_from_what_it_wrote(system, tmp_path):
    """The payoff: a run that stops itself resumes, where one killed at the wall does not.

    ⚠ The claim is that the *trajectory* continues — the iteration count carries across the
    interruption and the two runs land on the same energy — not that either converges. The
    tolerance here is 1e-10 Eh on a frozen-RDM surface, where the only thing that could
    differ is a piece of optimizer state that was not restored.
    """
    factors, h_ao, c0, spaces, solve = system
    kwargs = dict(e_nuc=0.0, conv_grad=1e-10, conv_energy=1e-12, report=False)

    straight = optimize_orbitals(factors, h_ao, c0, spaces, solve, max_iter=12, **kwargs)

    path = tmp_path / "run.chk"
    policy = CheckpointPolicy(path, min_interval=3600.0, cost_fraction=1e-12,
                              n_active_elec=2, deadline=Countdown(stop_after=3))
    optimize_orbitals(factors, h_ao, c0, spaces, solve, max_iter=12,
                      callback=policy.callback, **kwargs)
    checkpoint = read_checkpoint(path)
    assert checkpoint.iteration == 3

    resumed = optimize_orbitals(
        factors, h_ao, checkpoint.coeff, checkpoint.spaces, solve, max_iter=12,
        **dict(kwargs, **checkpoint.optimizer_kwargs(space_key=None)))
    assert resumed.n_iterations == straight.n_iterations
    assert resumed.energy == pytest.approx(straight.energy, abs=1e-10)


def test_without_a_checkpoint_the_run_still_stops_and_says_nothing_was_saved(system,
                                                                            kuiva_caplog):
    """⚠ Worth having anyway — the process exits cleanly and its output file is complete
    where a killed one ends mid-line — but the warning must not let anyone believe a
    restart point exists."""
    factors, h_ao, c0, spaces, solve = system
    deadline = Countdown(stop_after=1)
    result = optimize_orbitals(factors, h_ao, c0, spaces, solve, e_nuc=0.0, max_iter=20,
                               conv_grad=1e-10, conv_energy=1e-12, report=False,
                               callback=deadline.as_callback())
    assert result.n_iterations == 1 and not result.converged
    assert any("NO checkpoint" in r.getMessage() for r in kuiva_caplog.records)


def test_a_disabled_policy_still_carries_the_deadline(system, tmp_path):
    """``enabled=False`` turns the writing off, not the stopping."""
    factors, h_ao, c0, spaces, solve = system
    policy = CheckpointPolicy(tmp_path / "run.chk", enabled=False, n_active_elec=2,
                              deadline=Countdown(stop_after=2))
    result = optimize_orbitals(factors, h_ao, c0, spaces, solve, e_nuc=0.0, max_iter=20,
                               conv_grad=1e-10, conv_energy=1e-12, report=False,
                               callback=policy.callback)
    assert result.n_iterations == 2 and not result.converged
    assert not (tmp_path / "run.chk").exists()


def test_a_converged_iteration_is_not_a_deadline_stop(system, tmp_path):
    """⚠ A run that finished is not a run that ran out of time, however little was left.
    The distinction matters because "stopped by the deadline" is what tells a reader the
    result is an iterate rather than an answer."""
    factors, h_ao, c0, spaces, solve = system
    expired = Deadline(epoch=time.time() - 100.0, source="already past")
    grabbed = {}

    def grab(info):
        grabbed.update(info)
        return None

    optimize_orbitals(factors, h_ao, c0, spaces, solve, e_nuc=0.0, max_iter=1,
                      conv_grad=1e-10, conv_energy=1e-12, report=False, callback=grab)

    policy = CheckpointPolicy(tmp_path / "run.chk", min_interval=3600.0,
                              cost_fraction=1e-12, n_active_elec=2, deadline=expired)
    assert policy.callback(dict(grabbed, converged=False)) is False
    assert expired.fired

    fresh = Deadline(epoch=time.time() - 100.0, source="already past")
    policy = CheckpointPolicy(tmp_path / "converged.chk", min_interval=3600.0,
                              cost_fraction=1e-12, n_active_elec=2, deadline=fresh)
    assert policy.callback(dict(grabbed, converged=True)) is None
    assert not fresh.fired
    # ... and the converged iteration was written anyway, by its own rule
    assert (tmp_path / "converged.chk").exists()


def test_a_chained_callback_still_runs_and_can_stop_on_its_own(system, tmp_path):
    """The hook is single, so the deadline, the checkpoint policy and a user's callback all
    have to coexist on it."""
    factors, h_ao, c0, spaces, solve = system
    seen = []

    def chain(info):
        seen.append(info["iteration"])
        return None

    policy = CheckpointPolicy(tmp_path / "run.chk", min_interval=0.0, cost_fraction=1.0,
                              n_active_elec=2, deadline=Countdown(stop_after=2), chain=chain)
    optimize_orbitals(factors, h_ao, c0, spaces, solve, e_nuc=0.0, max_iter=20,
                      conv_grad=1e-10, conv_energy=1e-12, report=False,
                      callback=policy.callback)
    assert seen == [1, 2]
