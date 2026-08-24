"""Tests for HDF5 checkpointing and restart.

What each group can fail on:

* **round trip** — every field comes back bit for bit. Cheap, and it is the only thing that
  makes the rest meaningful.
* ⚠ **restart reproduces the uninterrupted trajectory, bitwise**, on a *frozen-RDM* problem.
  The freezing is the point: a re-solved CI is deterministic only to its residual tolerance
  (1e-8), so a run through it could only be compared to ~that, and a bitwise statement about
  the **optimizer state** would be untestable. With the RDMs fixed the surface is exactly one
  smooth function and any difference at all is a piece of state that was not restored.
* ⚠ **chart-scoped curvature** — a checkpoint written on one CI space, restored
  against another, must **clear** the L-BFGS pairs rather than restore them. Asserted through
  the warning as well as through the state, because the user needs to be told.
* **the adaptive budget** — thinning before skipping, and a ``WARNING`` on a skip. Asserted
  with ``kuiva_caplog``, since "proceeds but the user should know" is behaviour.
* **failure semantics**, which are the opposite of a cache's: a write failure warns
  and the run continues; a read failure on an explicit restart raises.
"""
import itertools

import numpy as np
import pytest

from kuiva.ci.strings import Determinants, hamiltonian_matrix, rdm12
from kuiva.integrals.transform import ThreeIndexAO
from kuiva.io.checkpoint import (CASSCFCheckpoint, CheckpointError, CheckpointPolicy,
                                 SCHEMA_VERSION, checkpoint_size_gb, code_fingerprint,
                                 read_checkpoint, write_checkpoint)
from kuiva.mcscf.casci import FullCISolver
from kuiva.mcscf.orbopt import (CASIntegrals, OrbitalOptimizer, OrbitalSpaces,
                                optimize_orbitals)

pytest.importorskip("h5py")


@pytest.fixture(scope="module")
def system():
    """The synthetic two-component system of ``test_mcscf``."""
    rng = np.random.default_rng(3)
    nao = 6
    n = 2 * nao
    npair = nao * (nao + 1) // 2
    factors = ThreeIndexAO(l_packed=rng.standard_normal((3 * nao, npair)), nao=nao,
                           origin="cholesky")
    h_ao = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    h_ao = 0.5 * (h_ao + h_ao.conj().T)
    _, c0 = np.linalg.eigh(h_ao)
    spaces = OrbitalSpaces.from_counts(n_inactive=2, n_active=4, n_orb=n)
    return factors, h_ao, np.ascontiguousarray(c0), spaces, 2


def exact_ci_solver(spaces, n_elec):
    """An exact CI over the active space, as the optimizer's callback."""
    dets = Determinants.from_occupations(
        itertools.combinations(range(spaces.n_active), n_elec), spaces.n_active)

    def solve(ints):
        matrix = hamiltonian_matrix(dets, ints.h_active_effective(), ints.active_eri())
        values, vectors = np.linalg.eigh(matrix.toarray())
        gamma, gamma2 = rdm12(dets, vectors[:, :1])
        return float(values[0]) + ints.e_core, gamma, gamma2

    return solve


def frozen_rdm_solver(system):
    """A **deterministic, frozen-RDM** solver: the energy is then one smooth function of the
    orbitals with no eigensolver noise in it at all, which is what lets a restart be compared
    bitwise rather than to a tolerance."""
    factors, h_ao, c0, spaces, n_elec = system
    ints = CASIntegrals.build(factors, h_ao, c0, spaces)
    dets = Determinants.from_occupations(
        itertools.combinations(range(spaces.n_active), n_elec), spaces.n_active)
    matrix = hamiltonian_matrix(dets, ints.h_active_effective(), ints.active_eri())
    _, vectors = np.linalg.eigh(matrix.toarray())
    gamma, gamma2 = rdm12(dets, vectors[:, :1])
    from kuiva.mcscf.orbopt import cas_energy

    def solve(ints_at):
        return cas_energy(ints_at, gamma, gamma2), gamma, gamma2

    return solve


def make_checkpoint(iteration=3, n_orb=8, n_active=4, n_states=2, ndet=6, with_ci=True):
    rng = np.random.default_rng(iteration)
    optimizer = OrbitalOptimizer(OrbitalSpaces.from_counts(2, n_active, n_orb))
    state = optimizer.state_dict(space_key="full-ci:4:2")
    state["lbfgs_s"] = (rng.standard_normal((2, optimizer.n_parameters))
                        + 1j * rng.standard_normal((2, optimizer.n_parameters)))
    state["lbfgs_y"] = np.array(state["lbfgs_s"])
    state["ah_guess"] = rng.standard_normal(optimizer.n_parameters) + 0j
    state["prev_energy"] = -1.25
    state["trust"] = 0.07
    return CASSCFCheckpoint(
        iteration=iteration, energy=-7.5, grad_norm=3.2e-5, converged=False,
        coeff=rng.standard_normal((2 * n_orb, n_orb)) + 1j * rng.standard_normal(
            (2 * n_orb, n_orb)),
        inactive=np.arange(2), active=np.arange(2, 2 + n_active),
        virtual=np.arange(2 + n_active, n_orb), n_orb=n_orb, n_active_elec=2,
        gamma=rng.standard_normal((n_active, n_active)) + 0j,
        gamma2=rng.standard_normal((n_active,) * 4) + 0j,
        state_energies=np.array([-7.6, -7.4][:n_states]),
        optimizer_state=state,
        ci_vectors=(rng.standard_normal((n_states, ndet)) + 0j) if with_ci else None,
        space_key="full-ci:4:2", history=np.array([-7.1, -7.4, -7.5]),
        metadata={"active_space": "the 4 lowest d spinors on Ti"})


# --- round trip ------------------------------------------------------------------------------

def test_a_checkpoint_round_trips_exactly(tmp_path):
    original = make_checkpoint()
    write_checkpoint(tmp_path / "run.chk", original)
    back = read_checkpoint(tmp_path / "run.chk")

    assert back.iteration == original.iteration
    assert back.energy == original.energy
    assert back.grad_norm == original.grad_norm
    assert back.converged == original.converged
    assert back.space_key == original.space_key
    assert back.metadata == original.metadata
    for name in ("coeff", "gamma", "gamma2", "state_energies", "history", "ci_vectors",
                 "inactive", "active", "virtual"):
        assert np.array_equal(getattr(back, name), getattr(original, name)), name
    for key, value in original.optimizer_state.items():
        restored = back.optimizer_state[key]
        if isinstance(value, np.ndarray):
            assert np.array_equal(restored, value), key
        else:
            assert restored == value or (value is None and restored is None), key


def test_the_spaces_come_back_as_a_validated_partition(tmp_path):
    original = make_checkpoint()
    write_checkpoint(tmp_path / "run.chk", original)
    spaces = read_checkpoint(tmp_path / "run.chk").spaces
    assert spaces.n_inactive == 2 and spaces.n_active == 4
    assert list(spaces.active) == [2, 3, 4, 5]


def test_a_thinned_checkpoint_has_no_ci_vectors_and_is_still_complete(tmp_path):
    thin = make_checkpoint().thinned()
    assert thin.ci_vectors is None
    write_checkpoint(tmp_path / "thin.chk", thin)
    back = read_checkpoint(tmp_path / "thin.chk")
    assert back.ci_vectors is None
    assert back.optimizer_state["trust"] == 0.07          # everything a restart needs is here


def test_the_sizing_function_is_exact_and_two_sided(tmp_path):
    """Sizing functions are exact and never pad. Bounded on both sides against the
    file that is actually written — HDF5's own structure is a few kB and does not scale."""
    checkpoint = make_checkpoint(n_orb=40, n_active=10, ndet=200)
    predicted = checkpoint_size_gb(checkpoint)
    assert predicted == sum(a.nbytes for a in checkpoint.payload()) / (1024.0 ** 3)
    actual = write_checkpoint(tmp_path / "big.chk", checkpoint)
    assert predicted <= actual                             # never optimistic
    assert actual - predicted < 0.001                      # and never padded: < 1 MB of HDF5


def test_the_write_is_atomic(tmp_path):
    """A run killed mid-write must leave the *previous* checkpoint, not a truncated file."""
    path = tmp_path / "run.chk"
    write_checkpoint(path, make_checkpoint(iteration=1))
    assert read_checkpoint(path).iteration == 1
    write_checkpoint(path, make_checkpoint(iteration=2))
    assert read_checkpoint(path).iteration == 2
    assert not list(tmp_path.glob("*.writing-*"))


# --- failure semantics (the opposite of a cache's) --------------------------------------

def test_a_missing_checkpoint_raises(tmp_path):
    with pytest.raises(CheckpointError, match="no checkpoint at"):
        read_checkpoint(tmp_path / "absent.chk")


def test_a_corrupt_checkpoint_raises(tmp_path):
    path = tmp_path / "junk.chk"
    path.write_bytes(b"this is not HDF5")
    with pytest.raises(CheckpointError, match="could not be read"):
        read_checkpoint(path)


def test_a_schema_mismatch_is_refused_not_guessed_at(tmp_path):
    import h5py

    path = tmp_path / "old.chk"
    write_checkpoint(path, make_checkpoint())
    with h5py.File(str(path), "r+") as handle:
        handle.attrs["schema_version"] = SCHEMA_VERSION + 1
    with pytest.raises(CheckpointError, match="schema version"):
        read_checkpoint(path)


def test_a_changed_code_fingerprint_only_warns(tmp_path, kuiva_caplog):
    """It says the sources moved, which is usually a comment and occasionally a reason not to
    trust the numbers — and only the reader can tell which, so it is not a refusal."""
    import h5py

    path = tmp_path / "other.chk"
    write_checkpoint(path, make_checkpoint())
    with h5py.File(str(path), "r+") as handle:
        handle.attrs["code_fingerprint"] = "0" * 32
    checkpoint = read_checkpoint(path)
    assert checkpoint.iteration == 3
    assert any("different build" in r.message for r in kuiva_caplog.records)
    assert code_fingerprint() != "0" * 32


def test_a_write_failure_warns_and_the_run_continues(tmp_path, kuiva_caplog):
    """⚠ Losing a restart point is not losing the calculation."""
    policy = CheckpointPolicy(tmp_path / "nowhere" / "run.chk")
    policy.path = tmp_path / "definitely" / "not" / "writable" / "run.chk"

    def explode(*args, **kwargs):
        raise OSError("disk full")

    import kuiva.io.checkpoint as module
    original = module.write_checkpoint
    module.write_checkpoint = explode
    try:
        assert policy.write(make_checkpoint(), force=True) is False
    finally:
        module.write_checkpoint = original
    assert any("could not write the checkpoint" in r.message for r in kuiva_caplog.records)
    assert policy.stats.n_skipped == 1


# --- the adaptive budget -----------------------------------------------------------------

def test_a_checkpoint_over_the_budget_is_thinned_before_it_is_skipped(tmp_path):
    checkpoint = make_checkpoint(n_orb=20, n_active=6, ndet=4000, n_states=2)
    thin_gb = checkpoint.thinned().size_gb()
    assert checkpoint.size_gb() > thin_gb
    policy = CheckpointPolicy(tmp_path / "run.chk",
                              budget_gb=0.5 * (checkpoint.size_gb() + thin_gb),
                              cost_fraction=1.0, min_interval=0.0)
    policy._last_write -= 100.0                # 100 s of compute to amortize the write over
    assert policy.write(checkpoint) is True
    assert policy.stats.n_thinned == 1
    assert read_checkpoint(tmp_path / "run.chk").ci_vectors is None


def test_a_checkpoint_costing_more_than_its_share_of_the_compute_is_skipped(
        tmp_path, kuiva_caplog):
    """The ~5% checkpoint-cost rule, against a *measured* bandwidth: a write that would eat the compute it
    is protecting is not worth doing, and the user is told so."""
    policy = CheckpointPolicy(tmp_path / "run.chk", budget_gb=100.0, min_interval=0.0,
                              cost_fraction=1e-12)
    policy._last_write -= 1.0
    assert policy.write(make_checkpoint()) is False
    assert policy.stats.n_skipped == 1
    assert not (tmp_path / "run.chk").exists()
    assert any("SKIPPED" in r.message for r in kuiva_caplog.records)


def test_a_checkpoint_over_the_budget_even_when_thin_is_skipped(tmp_path, kuiva_caplog):
    policy = CheckpointPolicy(tmp_path / "run.chk", budget_gb=1e-12, min_interval=0.0,
                              cost_fraction=1.0)
    policy._last_write -= 100.0
    assert policy.write(make_checkpoint()) is False
    assert policy.stats.n_skipped == 1
    assert not (tmp_path / "run.chk").exists()
    assert any("SKIPPED" in r.message for r in kuiva_caplog.records)


def test_the_minimum_interval_suppresses_a_checkpoint_quietly(tmp_path):
    """A cadence skip is expected behaviour, not something to warn about every iteration."""
    policy = CheckpointPolicy(tmp_path / "run.chk", min_interval=3600.0, cost_fraction=1.0)
    assert policy.write(make_checkpoint()) is False
    assert policy.stats.n_skipped == 1


def test_a_converged_iteration_is_written_whatever_the_cadence_says(tmp_path):
    """⚠ The last checkpoint is the *result*, not insurance against a crash."""
    policy = CheckpointPolicy(tmp_path / "run.chk", min_interval=3600.0, cost_fraction=1e-12)
    assert policy.write(make_checkpoint(), force=True) is True
    assert read_checkpoint(tmp_path / "run.chk").iteration == 3


def test_the_disk_bandwidth_probe_never_raises(tmp_path):
    from kuiva.util import resources as res

    measured = res.disk_write_bandwidth_gb_s(tmp_path)
    assert measured > 0.0
    assert res.disk_write_bandwidth_gb_s(tmp_path) == measured        # cached, probed once
    assert res.disk_write_bandwidth_gb_s("/proc/definitely/not/a/dir") > 0.0


def test_the_checkpoint_limits_come_from_the_resource_module():
    from kuiva.util import resources as res

    limits = res.ResourceLimits.from_config(4.0, checkpoint_budget_gb=1.5,
                                            checkpoint_min_interval_seconds=30.0)
    assert limits.checkpoint_budget_gb == 1.5
    assert limits.checkpoint_min_interval_seconds == 30.0
    with pytest.raises(res.ConfigurationError, match="checkpoint_cost_fraction"):
        res.ResourceLimits.from_config(4.0, checkpoint_cost_fraction=0.0)


# --- optimizer state -----------------------------------------------------------------

def test_the_optimizer_state_round_trips_through_the_object(system):
    factors, h_ao, c0, spaces, n_elec = system
    solver = exact_ci_solver(spaces, n_elec)
    ints = CASIntegrals.build(factors, h_ao, c0, spaces)
    energy, gamma, gamma2 = solver(ints)

    first = OrbitalOptimizer(spaces, mode="quasi-newton")
    step = first.step(ints, gamma, gamma2, factors, c0, energy=energy)
    grad, _ = first.gradient(ints, gamma, gamma2, factors, c0)
    first.accept(grad)

    second = OrbitalOptimizer(spaces, mode="quasi-newton")
    second.load_state_dict(first.state_dict(space_key="k"), space_key="k")
    assert second.trust == first.trust
    assert len(second._s) == len(first._s)
    next_step = second.step(ints, gamma, gamma2, factors, c0, energy=energy)
    assert np.array_equal(next_step.kappa, first.step(ints, gamma, gamma2, factors, c0,
                                                      energy=energy).kappa)
    del step


def test_a_different_parameter_count_is_refused(system):
    _, _, _, spaces, _ = system
    state = OrbitalOptimizer(spaces).state_dict()
    other = OrbitalOptimizer(OrbitalSpaces.from_counts(4, 4, spaces.n_orb))
    assert other.n_parameters != OrbitalOptimizer(spaces).n_parameters
    with pytest.raises(ValueError, match="rotation parameters"):
        other.load_state_dict(state)


def test_curvature_is_cleared_across_a_chart_change(system, kuiva_caplog):
    """⚠ Chart-scoped curvature: the L-BFGS pairs, the AH warm start and the trust radius are statements about
    an energy surface. Restoring them against a different CI space is precisely the bug
    chart-scoping exists to prevent."""
    _, _, _, spaces, _ = system
    source = OrbitalOptimizer(spaces, mode="quasi-newton")
    source._s = [np.ones(source.n_parameters, dtype=np.complex128)]
    source._y = [np.ones(source.n_parameters, dtype=np.complex128)]
    source._ah_guess = np.ones(source.n_parameters, dtype=np.complex128)
    source._stalls = 4
    source.trust = 1e-6
    state = source.state_dict(space_key="space-A")

    same = OrbitalOptimizer(spaces, mode="quasi-newton")
    same.load_state_dict(state, space_key="space-A")
    assert len(same._s) == 1 and same._ah_guess is not None and same._stalls == 4

    other = OrbitalOptimizer(spaces, mode="quasi-newton")
    other.load_state_dict(state, space_key="space-B")
    assert other._s == [] and other._y == []
    assert other._ah_guess is None
    assert other._stalls == 0
    assert other.trust > 1e-6                       # restored to a floor, not inherited
    assert any("no longer exists" in r.message for r in kuiva_caplog.records)


def test_an_absent_optional_comes_back_as_none_not_an_empty_array(tmp_path, system):
    """⚠ Otherwise a restored optimizer warm starts its augmented Hessian from a zero-length
    guess, which is not the same thing as from nothing."""
    _, _, _, spaces, _ = system
    fresh = OrbitalOptimizer(spaces)
    assert fresh.state_dict()["ah_guess"] is None
    checkpoint = make_checkpoint()
    checkpoint.optimizer_state["ah_guess"] = None
    checkpoint.optimizer_state["prev_grad"] = None
    write_checkpoint(tmp_path / "run.chk", checkpoint)
    back = read_checkpoint(tmp_path / "run.chk")
    assert back.optimizer_state["ah_guess"] is None
    assert back.optimizer_state["prev_grad"] is None


# --- restart (the point of all of the above) ----------------------------------------------------

@pytest.mark.parametrize("mode", ["second-order", "quasi-newton"])
def test_a_restart_reproduces_the_uninterrupted_trajectory_bitwise(system, tmp_path, mode):
    """⚠ The strongest statement available, and the frozen RDMs are what make it available.

    With the density matrices held fixed the objective is exactly one smooth deterministic
    function of the orbitals — no eigensolver, no residual tolerance, no noise floor — so an
    interrupted-and-restarted run and an uninterrupted one must agree to the **last bit**.
    Anything less is a piece of optimizer state that was not carried across, and a tolerance
    would let it hide. (A real CASSCF re-solves its CI to 1e-8, so it can only be compared to
    about that; that is the *CI's* tolerance, not the checkpoint's, and it is asserted
    separately below.)
    """
    factors, h_ao, c0, spaces, _ = system
    solver = frozen_rdm_solver(system)
    kwargs = dict(e_nuc=0.0, conv_grad=1e-10, conv_energy=1e-12, mode=mode, report=False)

    straight = optimize_orbitals(factors, h_ao, c0, spaces, solver, max_iter=8, **kwargs)

    path = tmp_path / "run.chk"
    policy = CheckpointPolicy(path, min_interval=0.0, cost_fraction=1.0, n_active_elec=2)
    optimize_orbitals(factors, h_ao, c0, spaces, solver, max_iter=4,
                      callback=policy.callback, **kwargs)
    checkpoint = read_checkpoint(path)
    assert checkpoint.iteration == 4

    resumed = optimize_orbitals(
        factors, h_ao, checkpoint.coeff, checkpoint.spaces, solver, max_iter=8,
        # A plain callable declares no chart, so both sides are None and chart-scoping is
        # unconditionally satisfied -- which is the documented behaviour for a wrapped
        # ci_solver, not an accident of this test.
        **dict(kwargs, **checkpoint.optimizer_kwargs(space_key=None)))
    assert resumed.n_iterations == straight.n_iterations
    assert resumed.energy == straight.energy                       # bitwise
    assert np.array_equal(resumed.coeff, straight.coeff)
    assert resumed.grad_norm == straight.grad_norm
    assert resumed.history == straight.history


def test_a_full_casscf_restart_converges_to_the_same_answer(system, tmp_path):
    """The same thing with the CI actually re-solved through :class:`FullCISolver`.

    ⚠ Deliberately **not** asserted as converged: the scheme is two-step, so on this
    stiff synthetic surface the neglected orbital-CI coupling parks the gradient around 5e-4
    long after the energy has stopped moving at 1e-9. That is a property of the optimizer and
    is measured in ``test_mcscf``; what is under test here is that an interrupted run and an
    uninterrupted one, given the same budget, end in the same place.
    """
    factors, h_ao, c0, spaces, n_elec = system
    common = dict(e_nuc=0.0, mode="second-order", conv_grad=1e-8, report=False)
    budget = 12

    straight = optimize_orbitals(factors, h_ao, c0, spaces, FullCISolver(
        spaces.n_active, n_elec, enforce_kramers=False), max_iter=budget, **common)

    path = tmp_path / "casscf.chk"
    solver = FullCISolver(spaces.n_active, n_elec, enforce_kramers=False)
    policy = CheckpointPolicy(path, solver=solver, min_interval=0.0, cost_fraction=1.0)
    policy._last_write -= 100.0
    optimize_orbitals(factors, h_ao, c0, spaces, solver, max_iter=budget // 3,
                      callback=policy.callback, **common)

    checkpoint = read_checkpoint(path)
    assert checkpoint.ci_vectors is not None                # the Davidson warm start survives
    assert checkpoint.space_key == "full-ci:4:2"
    assert checkpoint.state_energies.size == 1
    resumed_solver = FullCISolver(spaces.n_active, n_elec, enforce_kramers=False)
    resumed_solver.set_guess(checkpoint.ci_vectors)
    resumed = optimize_orbitals(
        factors, h_ao, checkpoint.coeff, checkpoint.spaces, resumed_solver, max_iter=budget,
        **dict(common,
               **checkpoint.optimizer_kwargs(space_key=resumed_solver.space_key())))
    assert resumed.n_iterations == straight.n_iterations == budget
    assert abs(resumed.energy - straight.energy) < 1e-9
    assert len(resumed.history) == len(straight.history)
    assert abs(resumed.grad_norm - straight.grad_norm) < 1e-6


def test_the_policy_writes_a_checkpoint_every_macro_iteration(system, tmp_path):
    factors, h_ao, c0, spaces, n_elec = system
    solver = FullCISolver(spaces.n_active, n_elec, enforce_kramers=False)
    policy = CheckpointPolicy(tmp_path / "run.chk", solver=solver, min_interval=0.0,
                              cost_fraction=1.0)
    optimize_orbitals(factors, h_ao, c0, spaces, solver, max_iter=5, e_nuc=0.0,
                      mode="second-order", callback=policy.callback, report=False)
    assert policy.stats.n_written == 5
    assert policy.stats.gb_written > 0.0
    final = read_checkpoint(tmp_path / "run.chk")
    assert final.iteration == 5
    # The state average is recorded by the policy itself, from the solver, so that a
    # restart can always be checked against it (test_a_restart_with_a_different_state_average
    # _is_refused is the mechanism test). Nothing else was asked for here.
    assert final.metadata == {"state_average": "n_states=1;weights=equal"}
    assert final.n_active_elec == n_elec


def test_a_user_callback_is_chained_not_replaced(system, tmp_path):
    """⚠ The optimizer has one hook, and a wall-clock budget has to share it."""
    factors, h_ao, c0, spaces, n_elec = system
    seen = []

    def budget(info):
        seen.append(info["iteration"])
        return False if info["iteration"] >= 2 else None

    solver = FullCISolver(spaces.n_active, n_elec, enforce_kramers=False)
    policy = CheckpointPolicy(tmp_path / "run.chk", solver=solver, min_interval=0.0,
                              cost_fraction=1.0, chain=budget)
    result = optimize_orbitals(factors, h_ao, c0, spaces, solver, max_iter=10, e_nuc=0.0,
                               mode="second-order", callback=policy.callback, report=False)
    assert seen == [1, 2]
    assert result.n_iterations == 2
    assert read_checkpoint(tmp_path / "run.chk").iteration == 2


# --- the restart's state average, and the chart it is compared against ---------------------

def test_optimizer_kwargs_carries_the_LIVE_key_not_the_files_own(system):
    """⚠ **The mechanism, not the observable.** Chart-scoping compares the key recorded
    *inside* ``optimizer_state`` against the key of the solver that is about to run. Handing
    the checkpoint its own ``space_key`` back compares the file with itself, so ``same_chart``
    is unconditionally true and curvature is restored across any chart change without a word
    — and the run then converges to a number nobody can tell apart from a good one.

    The argument is required rather than defaulted precisely so that the defect cannot come
    back by omission: this test would fail as a ``TypeError`` if it ever grew a default.
    """
    checkpoint = make_checkpoint()
    assert checkpoint.space_key == "full-ci:4:2"

    with pytest.raises(TypeError):
        checkpoint.optimizer_kwargs()                  # no live key -> no silent self-compare

    kwargs = checkpoint.optimizer_kwargs(space_key="full-ci:4:4")
    assert kwargs["space_key"] == "full-ci:4:4"        # the LIVE key, not "full-ci:4:2"
    assert kwargs["optimizer_state"]["space_key"] == "full-ci:4:2"   # the file's, untouched

    # And the two together are what load_state_dict actually compares.
    _, _, _, spaces, _ = system
    opt = OrbitalOptimizer(OrbitalSpaces.from_counts(2, 4, 8), mode="quasi-newton")
    opt.load_state_dict(kwargs["optimizer_state"], space_key=kwargs["space_key"])
    assert opt._s == [] and opt._ah_guess is None      # cleared, because the charts differ


def test_the_policy_records_the_state_average_from_the_solver(system, tmp_path):
    """Read from the solver rather than accepted from the caller, so a driver cannot fail to
    describe its own state average and leave a restart uncheckable."""
    from kuiva.io.checkpoint import STATE_AVERAGE_KEY, state_average_key

    factors, h_ao, c0, spaces, n_elec = system
    solver = FullCISolver(spaces.n_active, n_elec, n_states=2, weights=[0.75, 0.25],
                          enforce_kramers=False)
    assert state_average_key(solver) == "n_states=2;weights=0.75,0.25"

    policy = CheckpointPolicy(tmp_path / "run.chk", solver=solver, min_interval=0.0,
                              cost_fraction=1.0, metadata={"active_space": "d shell"})
    optimize_orbitals(factors, h_ao, c0, spaces, solver, max_iter=1, e_nuc=0.0,
                      mode="second-order", callback=policy.callback, report=False)
    stored = read_checkpoint(tmp_path / "run.chk").metadata
    assert stored[STATE_AVERAGE_KEY] == "n_states=2;weights=0.75,0.25"
    assert stored["active_space"] == "d shell"         # the caller's entries are untouched


@pytest.mark.parametrize("n_states, weights, why", [
    (3, None, "a different state count"),
    (2, [0.6, 0.4], "different weights at the same count"),
])
def test_a_restart_with_a_different_state_average_is_refused(system, n_states, weights, why):
    """⚠ A different state average is a different *calculation*, not a different chart.

    The energy functional itself changes, so the converged orbitals and energy do; restoring
    the old average's curvature onto it produces a trajectory that is an average of two
    averages, with a plausible number at the end and nothing in the output saying so. The
    active space is refused on the same grounds a few lines away in the same function.
    """
    from kuiva.interface.api import _check_restart_state_average

    _, _, _, spaces, n_elec = system
    written_by = FullCISolver(spaces.n_active, n_elec, n_states=2, enforce_kramers=False)
    resumed = make_checkpoint()
    resumed.metadata["state_average"] = state_average_key_of(written_by)

    now = FullCISolver(spaces.n_active, n_elec, n_states=n_states, weights=weights,
                       enforce_kramers=False)
    with pytest.raises(ValueError, match="different state average is a different calc"):
        _check_restart_state_average(resumed, now, "run.chk")

    # ...and the matching one passes, or the refusal would be vacuous.
    same = FullCISolver(spaces.n_active, n_elec, n_states=2, enforce_kramers=False)
    _check_restart_state_average(resumed, same, "run.chk")


def test_a_checkpoint_predating_the_state_average_warns_rather_than_passing(
        system, kuiva_caplog):
    """⚠ "Cannot be compared" is a weaker statement than "matches", and is said as such."""
    from kuiva.interface.api import _check_restart_state_average

    _, _, _, spaces, n_elec = system
    resumed = make_checkpoint()
    resumed.metadata.pop("state_average", None)
    _check_restart_state_average(resumed, FullCISolver(spaces.n_active, n_elec, n_states=5,
                                                       enforce_kramers=False), "old.chk")
    assert any("cannot be checked against it" in r.message for r in kuiva_caplog.records)


def state_average_key_of(solver):
    from kuiva.io.checkpoint import state_average_key
    return state_average_key(solver)


# --- a solver whose `last` is not a CASCIResult --------------------------------------------

def test_a_solver_without_ci_vectors_warns_instead_of_killing_the_run(system, tmp_path,
                                                                     kuiva_caplog):
    """⚠ **The write-failure promise, one step earlier.** ``from_info`` used to read
    ``last.vectors`` and ``last.total_energies`` unconditionally, and was called *outside* the
    guarded region — so a solver whose ``last`` is a tensor-network ``SweepResult`` (which has
    neither: its ``energies`` exclude ``e_core`` and are a different quantity) raised an
    ``AttributeError`` straight into the optimizer's loop and killed the calculation. The
    module's documented semantics are that losing a restart point is a WARNING and the run
    continues.
    """
    factors, h_ao, c0, spaces, n_elec = system

    class SweepLike(object):
        """Only what a SweepResult has: no `vectors`, no `total_energies`."""
        energies = np.array([-7.6, -7.4])
        max_discarded = 1e-9

    class NetworkLikeSolver(object):
        n_roots = 2
        requested_weights = None

        def __init__(self, inner):
            self._inner = inner
            self.last = SweepLike()

        def space_key(self):
            return "network:4:2"

        def __call__(self, ints):
            return self._inner(ints)

    solver = NetworkLikeSolver(frozen_rdm_solver(system))
    policy = CheckpointPolicy(tmp_path / "net.chk", solver=solver, min_interval=0.0,
                              cost_fraction=1.0)
    result = optimize_orbitals(factors, h_ao, c0, spaces, solver, max_iter=3, e_nuc=0.0,
                               mode="second-order", callback=policy.callback, report=False)

    assert result.n_iterations == 3                      # the run was NOT killed
    assert policy.stats.n_written == 3                   # and it still checkpointed
    back = read_checkpoint(tmp_path / "net.chk")
    assert back.ci_vectors is None                       # nothing invented
    assert back.state_energies.size == 0                 # local energies NOT passed off as totals
    assert back.space_key == "network:4:2"
    assert back.metadata["state_average"] == "n_states=2;weights=equal"   # n_roots, duck-typed


def test_an_exception_building_a_checkpoint_warns_and_the_run_continues(system, tmp_path,
                                                                       kuiva_caplog):
    """The guard is around ``from_info`` as well as around the write itself: an exception
    while *assembling* the object is the same promise broken one step earlier."""
    factors, h_ao, c0, spaces, n_elec = system
    solver = FullCISolver(spaces.n_active, n_elec, enforce_kramers=False)
    policy = CheckpointPolicy(tmp_path / "run.chk", solver=solver, min_interval=0.0,
                              cost_fraction=1.0)

    def explode(info):
        raise RuntimeError("no idea what a checkpoint is")

    policy.from_info = explode
    result = optimize_orbitals(factors, h_ao, c0, spaces, solver, max_iter=2, e_nuc=0.0,
                               mode="second-order", callback=policy.callback, report=False)
    assert result.n_iterations == 2
    assert policy.stats.n_written == 0 and policy.stats.n_skipped == 2
    assert any("could not assemble the checkpoint" in r.message
               for r in kuiva_caplog.records)
