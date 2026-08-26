"""Tier-0 tests for the environment pager: paging is a residence decision, never numerical.

Every comparison against the unpaged path is **bitwise**, because a page-in hands back
views into the very bytes that were written — anything less would mean the paged and
resident paths do not run the same arithmetic. The scratch-directory rule is asserted too:
paging is an escape hatch, and an escape hatch that cannot open reverts to the refusal the
budget was about to raise, never to a different failure.
"""
import gc
import os

import numpy as np
import pytest

from kuiva.dmrg.graph import NetworkGraph
from kuiva.dmrg.paging import EnvironmentPager
from kuiva.dmrg.sweep import EnvironmentCache, random_state
from kuiva.dmrg.ttno import compile_ttno, hamiltonian_product_terms
from kuiva.util import resources as res
from kuiva.util.scratch import ExtentFile

from test_ci_strings import random_spinor_integrals
from test_dmrg_sweep import exact_energies, solve


@pytest.fixture
def scratch_limits(monkeypatch, tmp_path):
    """Global limits with a test-owned scratch directory (no test touches site scratch)."""
    monkeypatch.setattr(res.BUDGET, "_limits",
                        res.ResourceLimits(memory_gb=8.0, source="test",
                                           scratch_dir=str(tmp_path)))
    return tmp_path


def _small_problem(n=4, k=2, max_bond=16, seed=5):
    h, eri = random_spinor_integrals(n, seed=seed)
    graph = NetworkGraph.path(n)
    op = compile_ttno(graph, hamiltonian_product_terms(h, eri))
    state = random_state(op, k, max_bond, rng=np.random.default_rng(seed))
    return op, state


# --- the ExtentFile allocator -------------------------------------------------------------


def test_extent_file_reuses_and_coalesces(tmp_path):
    """The point of extents over append-only: entries replaced every sweep must not grow
    the file by one generation per sweep."""
    f = ExtentFile(str(tmp_path / "extents.bin"))
    a = f.allocate(100)
    b = f.allocate(50)
    assert (a, b) == (0, 100) and f.size_gb == pytest.approx(150 / 1024.0 ** 3)
    f.free(a, 100)
    assert f.allocate(80) == 0                 # best-fit reuse of the hole
    f.free(0, 80)                              # coalesces with the (80, 20) remainder
    assert f.allocate(100) == 0
    f.free(0, 100)
    f.free(100, 50)                            # everything free: the file shrinks to zero
    assert f.size_gb == 0.0
    path = f.path
    del f
    gc.collect()
    assert not os.path.exists(path)


def test_extent_file_roundtrips_bytes(tmp_path):
    f = ExtentFile(str(tmp_path / "rt.bin"))
    rng = np.random.default_rng(0)
    a = rng.standard_normal(37) + 1j * rng.standard_normal(37)
    b = rng.standard_normal(11) + 1j * rng.standard_normal(11)
    off = f.allocate(a.nbytes + b.nbytes)
    f.write_at(off, [a, b])
    back = np.empty(48, dtype=np.complex128)
    f.read_at(off, back)
    np.testing.assert_array_equal(back[:37], a)
    np.testing.assert_array_equal(back[37:], b)


# --- page-out / page-in of real environments ----------------------------------------------


def test_environments_page_out_and_back_bitwise(scratch_limits):
    """A paged environment is the written environment: blocks, sectors, spaces, charge."""
    op, state = _small_problem()
    plain = EnvironmentCache(op, state)
    paged = EnvironmentCache(op, state, pager=EnvironmentPager(), resident_cap_gb=0.0)

    keys = [(3, 2), (2, 1), (1, 0)]            # subtrees pointing at the center (the root)
    reference = {kv: plain.get(*kv) for kv in keys}
    for kv in keys:
        paged.get(*kv)                         # cap 0: everything but the newest evicts
    assert paged._pager.n_stored > 0
    for kv in keys:
        env = paged.get(*kv)                   # most of these come back from scratch
        ref = reference[kv]
        assert env.spaces == ref.spaces and env.signs == ref.signs
        assert env.charge == ref.charge
        np.testing.assert_array_equal(env.sectors, ref.sectors)
        assert len(env.blocks) == len(ref.blocks)
        for got, want in zip(env.blocks, ref.blocks):
            np.testing.assert_array_equal(got, want)
    assert paged._pager.n_loaded > 0
    plain.release_all()
    paged.release_all()


def test_release_all_closes_the_pager_and_clears_the_ledger(scratch_limits):
    op, state = _small_problem()
    base = res.BUDGET.resident_gb()
    cache = EnvironmentCache(op, state, pager=EnvironmentPager(), resident_cap_gb=0.0)
    for kv in [(3, 2), (2, 1), (1, 0)]:
        cache.get(*kv)
    path = cache._pager._file.path
    assert os.path.exists(path)
    cache.release_all()
    assert res.BUDGET.resident_gb() == pytest.approx(base)
    assert not os.path.exists(path)


def test_a_fully_paged_solve_is_bitwise_the_unpaged_one(scratch_limits):
    """The whole claim, end to end: a solve forced to page every cold environment
    reproduces the unpaged solve's energies exactly, and still matches exact CI."""
    n, k = 5, 2
    h, eri = random_spinor_integrals(n, seed=11)
    graph = NetworkGraph.path(n)
    plain = solve(graph, h, eri, k, page_environments=False)
    paged = solve(graph, h, eri, k, environment_resident_gb=0.0)
    assert plain.converged and paged.converged
    np.testing.assert_array_equal(np.asarray(paged.energies), np.asarray(plain.energies))
    ref = exact_energies(n, k, h, eri, 1)
    assert abs(paged.energies[0] - ref[0]) < 1e-8


# --- the escape hatch cannot replace the refusal ------------------------------------------


def test_an_explicit_cap_without_scratch_refuses_with_the_configuration_error(monkeypatch):
    """A *requested* residence cap needs scratch: refusing with the configuration error —
    the one that names scratch_dir and $KUIVA_SCRATCH — is the honest answer."""
    monkeypatch.setattr(res.BUDGET, "_limits",
                        res.ResourceLimits(memory_gb=8.0, source="test"))
    op, state = _small_problem()
    cache = EnvironmentCache(op, state, pager=EnvironmentPager(), resident_cap_gb=0.0)
    cache.get(3, 2)                            # nothing older to evict: fine
    with pytest.raises(res.ConfigurationError, match="scratch"):
        cache.get(2, 1)                        # cap eviction must page (3,2) out: refuses
    cache.release_all()


def test_pressure_paging_that_cannot_page_reverts_to_the_memory_refusal(
        scratch_limits, monkeypatch, kuiva_caplog):
    """Pressure paging is an escape hatch: when the hatch fails (here: the store itself),
    the original MemoryLimitError stands — a calculation must never trade a diagnosed
    memory refusal for a scratch traceback it did not ask about."""
    op, state = _small_problem()
    cache = EnvironmentCache(op, state, pager=EnvironmentPager())
    env = cache.get(3, 2)
    env_gb = env.nbytes / 1024.0 ** 3
    # A limit the next environment cannot fit, so its reservation raises and the cache
    # tries to evict (3, 2) — whose page-out we make fail.
    monkeypatch.setattr(res.BUDGET, "_limits",
                        res.ResourceLimits(memory_gb=res.BUDGET.resident_gb()
                                           + 0.1 * env_gb, source="test",
                                           scratch_dir=str(scratch_limits)))

    def broken_store(self, key, tensor):
        raise res.ScratchLimitError("no room")

    monkeypatch.setattr(EnvironmentPager, "store", broken_store)
    with pytest.raises(res.MemoryLimitError):
        cache.get(2, 1)
    assert any("cannot" in r.getMessage() for r in kuiva_caplog.records)
    cache.release_all()
