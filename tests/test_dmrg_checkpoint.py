"""Tier-0 tests for the network-state checkpoint (``kuiva/dmrg/checkpoint.py``).

The claims, in order: the file round-trips the state **bitwise** (a warm start that is not
the state that was written is a different guess wearing its name); the sizing function is
exact and two-sided; the failure semantics follow the checkpoint rules (write failures warn,
read failures on an explicit restart raise, a schema mismatch is refused); the rolling
per-sweep policy writes what it says; and a restart is a warm start — same converged
energies, fewer sweeps — whose loss (wrong topology, wrong root count) degrades to a cold
start with a warning, while a wrong electron count is refused as a different calculation.
"""
import numpy as np
import pytest

from kuiva.dmrg.checkpoint import (NETWORK_SCHEMA_VERSION, NetworkCheckpointError,
                                   NetworkCheckpointPolicy, network_checkpoint_gb,
                                   network_state_path, read_network_state,
                                   write_network_state)
from kuiva.dmrg.graph import NetworkGraph
from kuiva.dmrg.solver import DMRGSolver
from kuiva.dmrg.sweep import random_state, solve_ttn
from kuiva.dmrg.ttno import compile_ttno, hamiltonian_product_terms

from test_ci_strings import random_spinor_integrals
from test_dmrg_sweep import exact_energies

pytest.importorskip("h5py")

E_TOL = 1e-8


class _Ints:
    """The duck-typed integrals object the ci_solver contract passes around."""

    def __init__(self, h, eri, e_core=0.0):
        self._h, self._eri, self.e_core = h, eri, float(e_core)

    def h_active_effective(self):
        return self._h

    def active_eri(self):
        return self._eri


def small_problem(n=6, k=2, seed=3):
    h, eri = random_spinor_integrals(n, seed=seed)
    graph = NetworkGraph.path(n)
    op = compile_ttno(graph, hamiltonian_product_terms(h, eri))
    return graph, op, h, eri, k


def assert_tensors_bitwise(ta, tb):
    assert ta.spaces == tb.spaces
    assert ta.signs == tb.signs
    assert ta.charge == tb.charge
    assert np.array_equal(ta.sectors, tb.sectors)
    for x, y in zip(ta.blocks, tb.blocks):
        assert np.array_equal(x, y)


def assert_states_bitwise(a, b):
    assert a.graph == b.graph
    assert a.center == b.center
    assert a.charge == b.charge
    assert a.charge.moduli == b.charge.moduli
    assert a.n_roots == b.n_roots
    for ta, tb in zip(a.tensors, b.tensors):
        if ta is None:
            assert tb is None
        else:
            assert_tensors_bitwise(ta, tb)
    for ca, cb in zip(a.centers, b.centers):
        assert_tensors_bitwise(ca, cb)


# --- the file ------------------------------------------------------------------------------

def test_round_trip_is_bitwise(tmp_path):
    graph, op, h, eri, k = small_problem()
    state = random_state(op, k, 12, n_roots=2, rng=np.random.default_rng(5))
    path = tmp_path / "state.h5"
    write_network_state(path, state, space_key="dmrg:test", sweep=7,
                        energies=[-1.0, -0.5], converged=True)
    read, meta = read_network_state(path)
    assert_states_bitwise(state, read)
    assert meta["space_key"] == "dmrg:test"
    assert meta["sweep"] == 7 and meta["converged"] is True
    assert meta["n_elec"] == k and meta["n_roots"] == 2
    assert np.array_equal(meta["energies"], [-1.0, -0.5])


def test_a_read_back_state_is_solvable(tmp_path):
    """The reconstructed tensors go through the validating constructor and then a sweep."""
    graph, op, h, eri, k = small_problem(seed=8)
    state = random_state(op, k, 200, n_roots=1, rng=np.random.default_rng(8))
    write_network_state(tmp_path / "s.h5", state)
    read, _ = read_network_state(tmp_path / "s.h5")
    result = solve_ttn(op, read, max_bond=200, boundary_check=0, report=False)
    assert result.converged
    ref = exact_energies(6, k, h, eri, 1)
    assert abs(result.energies[0] - ref[0]) < E_TOL


def test_sizing_is_exact_and_two_sided(tmp_path):
    graph, op, h, eri, k = small_problem()
    state = random_state(op, k, 12, n_roots=2, rng=np.random.default_rng(5))
    payload = sum(t.nbytes for t in state.tensors if t is not None)
    payload += sum(c.nbytes for c in state.centers)
    gb = network_checkpoint_gb(state)
    assert gb * 1024.0 ** 3 == payload                    # exact, never padded
    assert 0.0 < gb < 1.0


def test_a_missing_file_raises(tmp_path):
    with pytest.raises(NetworkCheckpointError, match="no network-state checkpoint"):
        read_network_state(tmp_path / "nothing.h5")


def test_a_corrupt_file_raises(tmp_path):
    path = tmp_path / "bad.h5"
    path.write_bytes(b"this is not an HDF5 file")
    with pytest.raises(NetworkCheckpointError, match="could not be read"):
        read_network_state(path)


def test_a_schema_mismatch_is_refused_not_guessed_at(tmp_path):
    import h5py

    graph, op, h, eri, k = small_problem()
    state = random_state(op, k, 8, n_roots=1, rng=np.random.default_rng(1))
    path = tmp_path / "s.h5"
    write_network_state(path, state)
    with h5py.File(str(path), "a") as handle:
        handle.attrs["network_schema_version"] = NETWORK_SCHEMA_VERSION + 1
    with pytest.raises(NetworkCheckpointError, match="schema version"):
        read_network_state(path)


def test_the_write_is_atomic(tmp_path):
    """A failed write leaves the previous file intact, and no temporary behind."""
    graph, op, h, eri, k = small_problem()
    state = random_state(op, k, 8, n_roots=1, rng=np.random.default_rng(1))
    path = tmp_path / "s.h5"
    write_network_state(path, state, sweep=1)
    before = path.read_bytes()
    other = random_state(op, k, 8, n_roots=1, rng=np.random.default_rng(2))
    import h5py
    real_file = h5py.File

    def broken(*args, **kwargs):
        raise OSError("disk on fire")

    h5py.File = broken
    try:
        with pytest.raises(OSError):
            write_network_state(path, other, sweep=2)
    finally:
        h5py.File = real_file
    assert path.read_bytes() == before
    assert list(tmp_path.glob("*.writing-*")) == []


def test_network_state_path_rule():
    assert str(network_state_path("run.h5")).endswith("run.network.h5")
    assert str(network_state_path("a/b/run.chk")).endswith("run.network.chk")
    assert str(network_state_path("run")).endswith("run.network.h5")


# --- the rolling policy through solve_ttn --------------------------------------------------

def test_the_policy_writes_every_completed_sweep_and_the_converged_state(tmp_path):
    graph, op, h, eri, k = small_problem(seed=21)
    state = random_state(op, k, 200, n_roots=2, rng=np.random.default_rng(21))
    policy = NetworkCheckpointPolicy(tmp_path / "roll.h5", min_interval=0.0,
                                     cost_fraction=1.0)
    result = solve_ttn(op, state, max_bond=200, boundary_check=0, report=False,
                       checkpoint=policy)
    assert result.converged
    assert policy.n_written == result.n_sweeps            # rolling: one write per sweep
    read, meta = read_network_state(tmp_path / "roll.h5")
    assert meta["converged"] is True                      # the last write is the result
    assert_states_bitwise(result.state, read)


def test_the_minimum_interval_suppresses_writes_but_never_the_converged_one(tmp_path):
    graph, op, h, eri, k = small_problem(seed=22)
    state = random_state(op, k, 200, n_roots=1, rng=np.random.default_rng(22))
    policy = NetworkCheckpointPolicy(tmp_path / "roll.h5", min_interval=3600.0,
                                     cost_fraction=1.0)
    result = solve_ttn(op, state, max_bond=200, boundary_check=0, report=False,
                       checkpoint=policy)
    assert result.converged
    assert policy.n_written == 1                          # only the unconditional one
    _, meta = read_network_state(tmp_path / "roll.h5")
    assert meta["converged"] is True


def test_a_write_failure_warns_and_the_sweep_continues(tmp_path, kuiva_caplog):
    graph, op, h, eri, k = small_problem(seed=23)
    state = random_state(op, k, 200, n_roots=1, rng=np.random.default_rng(23))
    policy = NetworkCheckpointPolicy(tmp_path / "no" / "dir" / "roll.h5",
                                     min_interval=0.0, cost_fraction=1.0)
    policy.path.parent.mkdir(parents=True)
    policy.path.parent.chmod(0o500)
    try:
        result = solve_ttn(op, state, max_bond=200, boundary_check=0, report=False,
                           checkpoint=policy)
    finally:
        policy.path.parent.chmod(0o700)
    assert result.converged                               # the run was never at risk
    assert policy.n_written == 0
    assert any("could not write the network state" in r.getMessage()
               for r in kuiva_caplog.records)


# --- the solver-level warm start -----------------------------------------------------------

def test_solver_checkpoint_and_restart_round_trip(tmp_path):
    graph, op, h, eri, k = small_problem(seed=31)
    ints = _Ints(h, eri, e_core=0.25)
    path = tmp_path / "net.h5"
    cold = DMRGSolver(k, max_bond=200, n_roots=2, graph=graph, enforce_kramers=False,
                      checkpoint=NetworkCheckpointPolicy(path, min_interval=0.0,
                                                         cost_fraction=1.0))
    e_cold, _, _ = cold.solve(ints)
    assert path.is_file()
    _, meta = read_network_state(path)
    assert meta["space_key"] == cold.space_key()

    warm = DMRGSolver(k, max_bond=200, n_roots=2, graph=graph, enforce_kramers=False,
                      restart=path)
    e_warm, _, _ = warm.solve(ints)
    assert abs(e_warm - e_cold) < E_TOL
    # a warm start from the converged state re-derives the fixed point in the minimum
    # number of sweeps (one to establish the energy, one to see it stationary)
    assert warm.last.n_sweeps <= 2


def test_restart_with_a_different_electron_count_is_refused(tmp_path):
    graph, op, h, eri, k = small_problem(seed=32)
    state = random_state(op, k, 8, n_roots=1, rng=np.random.default_rng(32))
    path = tmp_path / "net.h5"
    write_network_state(path, state)
    with pytest.raises(ValueError, match="active electrons"):
        DMRGSolver(k + 2, max_bond=8, graph=graph, restart=path)


def test_restart_with_a_different_root_count_warns_and_starts_cold(tmp_path, kuiva_caplog):
    graph, op, h, eri, k = small_problem(seed=33)
    state = random_state(op, k, 8, n_roots=1, rng=np.random.default_rng(33))
    path = tmp_path / "net.h5"
    write_network_state(path, state)
    solver = DMRGSolver(k, max_bond=8, n_roots=3, graph=graph, restart=path)
    assert solver._state is None                          # cold, not silently reshaped
    assert any("starts cold" in r.getMessage() for r in kuiva_caplog.records)


def test_restart_on_a_different_topology_warns_and_keeps_the_given_graph(tmp_path,
                                                                         kuiva_caplog):
    graph, op, h, eri, k = small_problem(seed=34)
    state = random_state(op, k, 8, n_roots=1, rng=np.random.default_rng(34))
    path = tmp_path / "net.h5"
    write_network_state(path, state)
    tree = NetworkGraph(6, [(0, 1), (1, 2), (1, 3), (3, 4), (3, 5)])
    solver = DMRGSolver(k, max_bond=8, n_roots=1, graph=tree, restart=path)
    assert solver._state is None
    assert solver.graph == tree
    assert any("different topology" in r.getMessage() for r in kuiva_caplog.records)


def test_restart_and_initial_state_are_mutually_exclusive(tmp_path):
    graph, op, h, eri, k = small_problem(seed=35)
    state = random_state(op, k, 8, n_roots=1, rng=np.random.default_rng(35))
    path = tmp_path / "net.h5"
    write_network_state(path, state)
    with pytest.raises(ValueError, match="one or the other"):
        DMRGSolver(k, max_bond=8, graph=graph, initial_state=state, restart=path)
