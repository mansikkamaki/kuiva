"""Tier-0 tests for the bond-dimension series and the ``E(w_disc -> 0)`` extrapolation.

The oracle is the dense CI spectrum of the same integrals. What is pinned: an exact series
is reported as exact rather than fitted (fitting rounding noise would manufacture a
spurious correction); a truncated series is variational down the caps and its extrapolate
lands closer to the exact energy than the largest cap alone — the property the protocol
exists for; and the validation refusals (too few caps, a descending series) fire, because a
descending step would truncate a converged warm start and call the damage a data point.
"""
import numpy as np
import pytest

from kuiva.dmrg.extrapolate import bond_series
from kuiva.dmrg.graph import NetworkGraph
from kuiva.dmrg.ttno import compile_ttno, hamiltonian_product_terms

from test_ci_strings import random_spinor_integrals
from test_dmrg_sweep import exact_energies


def problem(n, k, seed):
    h, eri = random_spinor_integrals(n, seed=seed)
    op = compile_ttno(NetworkGraph.path(n), hamiltonian_product_terms(h, eri))
    return op, h, eri


def test_an_exact_series_is_reported_exact_not_fitted():
    n, k = 6, 2
    op, h, eri = problem(n, k, seed=61)
    series = bond_series(op, k, [50, 200], report=False)
    assert series.exact
    assert float(series.discarded.max()) == 0.0
    assert np.max(np.abs(series.extrapolated - series.energies[-1])) == 0.0
    assert abs(series.sa_extrapolated - exact_energies(n, k, h, eri, 1)[0]) < 1e-8


def test_the_extrapolate_improves_on_the_largest_cap():
    n, k = 8, 4
    op, h, eri = problem(n, k, seed=63)
    exact = exact_energies(n, k, h, eri, 1)[0]
    series = bond_series(op, k, [8, 10, 12, 14], report=False)
    assert not series.exact
    assert np.all(np.diff(series.sa_energies) <= 1e-9)   # variational down the series
    err_last = abs(float(series.sa_energies[-1]) - exact)
    err_extra = abs(series.sa_extrapolated - exact)
    assert err_extra < err_last
    # the report carries the series beside the extrapolate; both live on the result
    assert series.residuals.shape == (1,) and np.all(np.isfinite(series.residuals))


def test_series_validation():
    n, k = 6, 2
    op, h, eri = problem(n, k, seed=63)
    with pytest.raises(ValueError, match="at least two"):
        bond_series(op, k, [8], report=False)
    with pytest.raises(ValueError, match="ascending"):
        bond_series(op, k, [16, 8], report=False)
