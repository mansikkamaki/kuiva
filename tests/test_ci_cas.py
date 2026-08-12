"""Tier-0 tests for complete-CAS addressing and the single-excitation map.

The whole full CI is built on two objects: a bijection between determinants and
``[0, C(n,k))``, and the two rectangular tables holding ``<K|E_pq|J>``. Both are pure integer
bookkeeping, so an error in either is silent — a wrong rank permutes the CI vector and every
norm survives it; a wrong sign flips a matrix element and leaves the Hamiltonian Hermitian and
plausible.

So the checks are chosen for what they can *fail on*:

* the mask ordering against **``itertools.combinations``** — a different construction entirely;
* rank/unrank round trips over the whole space, both directions;
* the excitation map against :func:`kuiva.ci.strings.single_excitation_operator`, which builds
  ``a_p^dag a_q`` from the ``O(N^2)`` pairwise connection search and its own independent phase
  routine. That comparison is element by element, including signs, and it is the one that
  would actually fail on a bad transposition or a dropped fermionic phase;
* the incidence identity ``C(n,k) k = C(n,k-1) (n-k+1)``, which is what makes the two tables
  transposes of one another rather than two independent constructions;
* the sizing functions two-sidedly against real arrays' ``nbytes`` (a sizing function
  that grew a safety factor must fail).
"""
import itertools
import math

import numpy as np
import pytest

from kuiva.ci import strings as st
from kuiva.ci.strings import (CASSpace, binomial_table, cas_dimension, cas_vector_gb,
                              excitation_map_gb, single_excitation_operator)

# (n_spinor, n_elec) pairs spanning the corners: one electron, one hole, half filling, a
# completely full space, and a space wide enough that a block boundary is exercised.
SPACES = [(4, 1), (5, 2), (6, 3), (6, 5), (7, 2), (8, 4), (9, 3), (5, 5), (10, 2)]


# --- the Pascal table --------------------------------------------------------------------

def test_binomial_table_is_exact():
    table = binomial_table(24, 12)
    for n in range(25):
        for k in range(13):
            assert table[n, k] == math.comb(n, k)


def test_binomial_table_survives_the_mask_limit():
    """C(64, 32) = 1.83e18 must be exact in int64, not silently wrapped."""
    table = binomial_table(64, 32)
    assert table[64, 32] == math.comb(64, 32)
    assert table.dtype == np.int64


# --- addressing --------------------------------------------------------------------------

@pytest.mark.parametrize("n,k", SPACES)
def test_masks_are_every_subset_in_ascending_order(n, k):
    """Independent construction: itertools, sorted. Colex rank *is* ascending mask order."""
    space = CASSpace(n, k, build_map=False)
    expected = np.array(sorted(sum(1 << p for p in combo)
                               for combo in itertools.combinations(range(n), k)),
                        dtype=np.uint64)
    assert space.ndet == cas_dimension(n, k) == math.comb(n, k)
    assert np.array_equal(space.masks, expected)


@pytest.mark.parametrize("n,k", SPACES)
def test_rank_and_unrank_are_inverse_bijections(n, k):
    space = CASSpace(n, k, build_map=False)
    ranks = np.arange(space.ndet, dtype=np.int64)
    assert np.array_equal(space.rank(space.masks), ranks)
    assert np.array_equal(space.unrank(ranks), space.masks)
    # Every determinant carries exactly k electrons, and no mask repeats.
    assert len(set(space.masks.tolist())) == space.ndet
    assert np.all(st._popcount(space.masks) == k)


def test_hole_strings_are_the_k_minus_one_space():
    space = CASSpace(7, 3)
    holes = space.hole_masks()
    assert holes.size == space.n_hole == math.comb(7, 2)
    expected = np.array(sorted(sum(1 << p for p in combo)
                               for combo in itertools.combinations(range(7), 2)),
                        dtype=np.uint64)
    assert np.array_equal(holes, expected)


def test_cas_space_refuses_impossible_occupations():
    with pytest.raises(ValueError):
        CASSpace(4, 5)
    with pytest.raises(ValueError):
        CASSpace(4, 0)
    with pytest.raises(ValueError):
        CASSpace(65, 2)


# --- the excitation map ------------------------------------------------------------------

def _map_operators(space):
    """``M[p, q, K, J] = <K|E_pq|J>`` rebuilt from the two tables, by their own contract."""
    n, k = space.n_spinor, space.n_elec
    ops = np.zeros((n, n, space.ndet, space.ndet))
    for det in range(space.ndet):
        for slot in range(k):
            hole = space.d2h_hole[det, slot]
            p = space.d2h_orb[det, slot]
            s1 = space.d2h_sign[det, slot]
            for j in range(space.n_empty):
                ops[p, space.h2d_orb[hole, j], det, space.h2d_det[hole, j]] += (
                    s1 * space.h2d_sign[hole, j])
    return ops


@pytest.mark.parametrize("n,k", [(4, 1), (5, 2), (6, 3), (6, 5), (7, 2)])
def test_map_reproduces_the_single_excitation_operators(n, k):
    """The check that can actually fail: against the independent ``O(N^2)`` search.

    :func:`single_excitation_operator` finds interacting pairs by XOR/popcount over all
    determinant pairs and signs them with :func:`kuiva.ci.strings._phase_single`. The map
    reaches the same elements by a completely different route — annihilate, index a hole
    string, create — so agreement element by element, signs included, is real evidence.
    """
    space = CASSpace(n, k)
    dets = space.determinants()
    built = _map_operators(space)
    for p in range(n):
        for q in range(n):
            reference = single_excitation_operator(dets, p, q).toarray()
            assert np.allclose(built[p, q], reference.real, atol=0)
            assert np.allclose(reference.imag, 0.0, atol=0)


@pytest.mark.parametrize("n,k", SPACES)
def test_map_tables_are_transposes_with_the_promised_shapes(n, k):
    space = CASSpace(n, k)
    assert space.h2d_det.shape == (space.n_hole, space.n_empty) == (math.comb(n, k - 1),
                                                                   n - k + 1)
    assert space.d2h_hole.shape == (space.ndet, k)
    # The identity that makes one construction serve both tables.
    assert space.ndet * k == space.n_hole * space.n_empty
    assert space.h2d_det.dtype == space.d2h_hole.dtype == np.int32
    for arr in (space.h2d_orb, space.h2d_sign, space.d2h_orb, space.d2h_sign):
        assert arr.dtype == np.int8
        assert arr.flags.c_contiguous
    # Every entry written: the transposition covers each (determinant, slot) exactly once.
    assert np.all(space.d2h_orb >= 0)
    assert np.all(np.abs(space.h2d_sign) == 1)
    assert np.all(np.abs(space.d2h_sign) == 1)


@pytest.mark.parametrize("n,k", SPACES)
def test_map_slots_are_the_ascending_occupied_and_empty_orbitals(n, k):
    """The layout contract, checked rather than assumed: ascending within every row."""
    space = CASSpace(n, k)
    occ = space.occupations()
    for det in range(space.ndet):
        orbitals = space.d2h_orb[det].astype(int)
        assert list(orbitals) == sorted(np.nonzero(occ[det])[0].tolist())
    hole_occ = st.occupation_matrix(space.hole_masks(), n)
    for hole in range(space.n_hole):
        empties = space.h2d_orb[hole].astype(int)
        assert list(empties) == sorted(np.nonzero(~hole_occ[hole])[0].tolist())


def test_map_signs_match_the_independent_phase_routine():
    """``h2d_sign`` against ``_phase_single`` applied to the same excitation."""
    space = CASSpace(6, 3)
    holes = space.hole_masks()
    for hole in range(space.n_hole):
        for j in range(space.n_empty):
            q = int(space.h2d_orb[hole, j])
            det = int(space.h2d_det[hole, j])
            # a_q^dag |hole> and a_q |det>: the same phase, which is what makes the two
            # tables shareable. _phase_single(det, q, q) = (-1)^(occupied below q, twice).
            below = st._popcount(np.array([holes[hole] & st._BELOW[q]], dtype=np.uint64))[0]
            assert space.h2d_sign[hole, j] == (-1) ** below
            slot = int(np.nonzero(space.d2h_orb[det] == q)[0][0])
            assert space.d2h_sign[det, slot] == space.h2d_sign[hole, j]
            assert space.d2h_hole[det, slot] == hole


# --- sizing -----------------------------------------------------------------------

@pytest.mark.parametrize("n,k", [(6, 3), (10, 5), (12, 6)])
def test_excitation_map_sizing_is_exact_two_sided(n, k):
    """⚠ Two-sided: a safety factor creeping into the estimate must fail the test."""
    space = CASSpace(n, k)
    actual = sum(a.nbytes for a in space.excitation_arrays())
    assert excitation_map_gb(n, k) == pytest.approx(actual / 1024.0 ** 3, rel=1e-12)


def test_cas_vector_sizing_is_exact_two_sided():
    vectors = np.zeros((cas_dimension(10, 5), 3), dtype=np.complex128)
    assert cas_vector_gb(10, 5, 3) == pytest.approx(vectors.nbytes / 1024.0 ** 3, rel=1e-12)


def test_map_reservation_refuses_before_allocating(monkeypatch):
    """A space that cannot fit is refused with advice naming the knob."""
    from kuiva.util import resources as res

    monkeypatch.setattr(res.BUDGET, "_limits",
                        res.ResourceLimits(memory_gb=1e-4, source="test"))
    with pytest.raises(res.MemoryLimitError) as excinfo:
        CASSpace(16, 8)
    assert "excitation map" in str(excinfo.value).lower()
    assert "dmrg" in str(excinfo.value).lower()
