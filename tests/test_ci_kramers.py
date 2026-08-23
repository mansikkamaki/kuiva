"""Tier-0/Tier-1 tests for the Kramers-restricted (time-reversal-adapted) CI.

What each group can actually fail on, which is the test of a check's worth:

* the **sign algebra** against an explicit operator string — ``T`` applied one creation
  operator at a time and bubble-sorted back to ascending order, which shares no line of code
  with the popcount arithmetic under test. A wrong sign there is invisible to every norm,
  trace and hermiticity check and shows up only as a Hamiltonian that stops commuting with
  ``T``, so it is asserted against a genuinely independent derivation;
* ``[H, T] = 0`` **as a computed property of the sigma operator**, together with a control
  that fails it — a random Hermitian spinor Hamiltonian is not time-reversal symmetric, and
  the check must say so, or it is measuring nothing;
* the **restricted path against the general one**, on the same integrals, in the same
  convention: energies, RDMs and a phase-invariant reduction of the transition densities.
  The general path is the reference path and this is the only kind of comparison that can
  fail on the new code alone;
* the **refusals** — an even electron count, an odd state count, an un-Kramers-paired orbital
  set — because each of those otherwise produces a converged, degenerate, plausible, wrong
  spectrum, which is the whole reason the mode is not the default;
* the **factor of two**, asserted as a strict inequality on applications of ``H`` rather than
  as a timing. It is the only claim the mode is made for, and it is the one thing a
  correctness test can pin without a stopwatch — and it is asserted **with its boundary**,
  because at two Kramers pairs it does not exist;
* the **whole pipeline downstream of the CI**, once per mode on a real atom, ending in
  SC-NEVPT2 — the sharpest of the consumers, because it rebuilds the determinant space on the
  ``N±1`` and ``N±2`` sectors where ``T²`` alternates sign and the restricted construction does
  not exist at all. That is the claim that the mode is an option and not a second pipeline.
"""
import numpy as np
import pytest

from kuiva.ci.davidson import davidson, davidson_kramers
from kuiva.ci.sigma import (SigmaOperator, assert_time_reversal, time_reversal_violation)
from kuiva.ci.strings import (CASSpace, diagonal_energies, kramers_partner,
                              kramers_representative, kramers_sign)
from kuiva.integrals.transform import ThreeIndexAO
from kuiva.mcscf.casci import FullCISolver, casscf
from kuiva.mcscf.orbopt import CASIntegrals, OrbitalSpaces
from kuiva.props.multiplet import block_moment_tensor
from kuiva.spinor.expand import expand_scalar_mos, two_component_operator
from test_ci_strings import random_spinor_integrals

#: Energy agreement demanded between the two symmetry modes. Well inside the Davidson residual
#: tolerance's second-order effect on a Ritz value, and far below anything physical.
ENERGY_TOL = 1e-12
#: The same for the state-averaged density matrices. Looser than the energies on purpose: they
#: are first order in the eigenvector error (see :mod:`kuiva.ci.davidson`), and the two paths
#: choose different — equally valid — bases inside each degenerate pair, so what is being
#: compared is an invariant of the pair reached by two routes.
RDM_TOL = 1e-10


# --- Integral sets ---------------------------------------------------------------------------

def time_reversal_symmetric_integrals(n, seed=7, spread=1.0, coupling=0.1):
    """Hermitian ``h`` and 4-fold ``eri``, **projected** onto their time-reversal-even part.

    ``h_pq -> (h_pq + t_p t_q conj(h_{pbar qbar})) / 2`` with ``t_k = (-1)^k``, and the
    analogous four-index projection. It is a projector, it commutes with hermiticity and with
    both 4-fold relations (all three are asserted below), and it produces a genuinely
    **two-component** Hamiltonian rather than a spin-free one — so the states it gives are
    Kramers pairs and not accidental spin multiplets.

    ``spread`` adds a Fock-like diagonal, which :mod:`kuiva.ci.davidson` requires of any
    large-space test integral set: a random Hermitian matrix is not a hard instance of the
    Davidson problem, it is a different problem.
    """
    rng = np.random.default_rng(seed)
    swap = np.arange(n) ^ 1
    t = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    sign4 = (t[:, None, None, None] * t[None, :, None, None]
             * t[None, None, :, None] * t[None, None, None, :])

    a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    h = 0.5 * (a + a.conj().T) + np.diag(spread * np.arange(n))
    h = 0.5 * (h + np.outer(t, t) * np.conj(h[np.ix_(swap, swap)]))

    b = rng.standard_normal((n * n, n * n)) + 1j * rng.standard_normal((n * n, n * n))
    eri = (coupling * (b + b.T)).reshape(n, n, n, n)
    eri = 0.5 * (eri + eri.transpose(1, 0, 3, 2).conj())
    eri = 0.5 * (eri + sign4 * np.conj(eri[np.ix_(swap, swap, swap, swap)]))
    return np.ascontiguousarray(h), np.ascontiguousarray(eri)


def kramers_paired_system(nao=8, n_inactive=2, n_active=10, seed=5, soc=0.15):
    """A synthetic molecule whose spinors are **exactly** Kramers paired.

    Spin-free three-index factors, a time-reversal-even two-component one-electron operator
    (:func:`kuiva.spinor.expand.two_component_operator` is that projection by construction),
    and a spinor set expanded from a *real* scalar orbital set — which is the guess the
    front-end produces and the one thing that makes an active space time-reversal symmetric.
    """
    rng = np.random.default_rng(seed)
    npair = nao * (nao + 1) // 2
    factors = ThreeIndexAO(l_packed=rng.standard_normal((3 * nao, npair)), nao=nao,
                           origin="cholesky")
    a_sf = rng.standard_normal((nao, nao))
    a_sf = 0.5 * (a_sf + a_sf.T)
    w = rng.standard_normal((3, nao, nao)) * soc
    w = 0.5 * (w - w.transpose(0, 2, 1))
    h_ao = two_component_operator(a_sf, w)
    q, _ = np.linalg.qr(rng.standard_normal((nao, nao)))
    coeff = np.ascontiguousarray(expand_scalar_mos(q).c)
    spaces = OrbitalSpaces.from_counts(n_inactive=n_inactive, n_active=n_active,
                                       n_orb=2 * nao)
    return factors, h_ao, coeff, spaces


@pytest.fixture(scope="module")
def paired_active_integrals():
    """``(h, eri, e_core)`` of a CAS(5, 10) over an exactly Kramers-paired orbital set."""
    factors, h_ao, coeff, spaces = kramers_paired_system()
    ints = CASIntegrals.build(factors, h_ao, coeff, spaces, e_nuc=0.0)
    return (np.ascontiguousarray(ints.h_active_effective()), ints.active_eri(),
            float(ints.e_core))


# --- K0: the time-reversal determinant algebra ----------------------------------------------

def brute_time_reverse(mask, n_spinor):
    """``(mask, sign)`` of ``T|I>`` by explicit operator strings — no popcount anywhere.

    ⚠ Deliberately the slow, obvious derivation: walk the occupied spinors in ascending order
    applying ``T a+_k T^-1 = (-1)^k a+_{k^1}``, then bubble-sort the resulting creation string
    back to ascending order counting transpositions. It shares nothing with
    :func:`kuiva.ci.strings.kramers_sign` except the convention it is testing.
    """
    operators = []
    sign = 1
    for k in range(n_spinor):
        if mask >> k & 1:
            sign *= 1 if k % 2 == 0 else -1
            operators.append(k ^ 1)
    for _ in range(len(operators)):
        for i in range(len(operators) - 1):
            if operators[i] > operators[i + 1]:
                operators[i], operators[i + 1] = operators[i + 1], operators[i]
                sign = -sign
    out = 0
    for k in operators:
        out |= 1 << k
    return out, sign


@pytest.mark.parametrize("n,k", [(4, 1), (4, 2), (4, 3), (6, 2), (6, 3), (8, 3), (8, 4),
                                 (8, 5), (10, 3)])
def test_the_time_reversal_sign_matches_an_explicit_operator_string(n, k):
    space = CASSpace(n, k, build_map=False)
    signs = kramers_sign(space.masks)
    partners = kramers_partner(space.masks)
    for index, mask in enumerate(space.masks):
        expected_mask, expected_sign = brute_time_reverse(int(mask), n)
        assert int(partners[index]) == expected_mask
        assert int(signs[index]) == expected_sign


@pytest.mark.parametrize("n,k", [(6, 3), (8, 4), (8, 5), (10, 3)])
def test_t_squared_is_minus_one_for_an_odd_electron_count(n, k):
    """Kramers' theorem in its determinant form, asserted as sign algebra."""
    space = CASSpace(n, k, build_map=False)
    kramers = space.kramers()
    expected = -1 if k % 2 else 1
    assert kramers.parity == expected

    rng = np.random.default_rng(0)
    vector = rng.standard_normal(space.ndet) + 1j * rng.standard_normal(space.ndet)
    twice = kramers.time_reverse(kramers.time_reverse(vector))
    assert np.array_equal(twice, expected * vector), "T^2 must be exact, not approximate"


@pytest.mark.parametrize("n,k", [(6, 3), (8, 5), (10, 3), (12, 5)])
def test_an_odd_electron_count_has_exactly_half_the_determinants_as_representatives(n, k):
    """No determinant is self-conjugate when ``T^2 = -1``, so the pairs tile the space."""
    space = CASSpace(n, k, build_map=False)
    kramers = space.kramers()
    assert kramers.n_pairs * 2 == space.ndet
    assert not np.any(kramers.partner == np.arange(space.ndet))
    # ... and the representative predicate is the "lowest singly occupied pair is unbarred"
    # rule, which is the same thing as the smaller mask.
    flags = kramers_representative(space.masks)
    assert np.array_equal(np.nonzero(flags)[0], kramers.representatives)


@pytest.mark.parametrize("n,k", [(6, 2), (8, 4)])
def test_an_even_electron_count_has_self_conjugate_determinants(n, k):
    """The other parity, asserted because it is a *different theorem* and not a special case.

    An all-closed determinant is its own time reverse, so the space is not tiled by pairs and
    nothing the restricted path is built on holds.
    """
    space = CASSpace(n, k, build_map=False)
    kramers = space.kramers()
    assert kramers.parity == 1
    assert np.any(kramers.partner == np.arange(space.ndet))
    assert kramers.n_pairs * 2 > space.ndet


@pytest.mark.parametrize("n,k", [(6, 3), (8, 5)])
def test_time_reversal_is_antiunitary_on_ci_vectors(n, k):
    """``<Ta|Tb> = <b|a>`` and ``T(c v) = conj(c) T v`` — the two properties every consumer
    of :meth:`~kuiva.ci.strings.KramersMap.time_reverse` relies on."""
    kramers = CASSpace(n, k, build_map=False).kramers()
    rng = np.random.default_rng(4)
    ndet = kramers.ndet
    a = rng.standard_normal(ndet) + 1j * rng.standard_normal(ndet)
    b = rng.standard_normal(ndet) + 1j * rng.standard_normal(ndet)
    scalar = 0.3 - 1.7j
    assert np.vdot(kramers.time_reverse(a), kramers.time_reverse(b)) == pytest.approx(
        np.vdot(b, a), abs=1e-12)
    assert np.allclose(kramers.time_reverse(scalar * a),
                       np.conj(scalar) * kramers.time_reverse(a), atol=1e-14)


# --- The precondition: [H, T] = 0, and a control that fails it -------------------------------

@pytest.mark.parametrize("n,k", [(6, 3), (8, 3), (8, 5), (8, 4)])
def test_the_sigma_operator_commutes_with_time_reversal(n, k):
    """The whole restricted path in one line, checked on the operator rather than assumed."""
    space = CASSpace(n, k)
    h, eri = time_reversal_symmetric_integrals(n)
    sigma = SigmaOperator(space, h, eri)
    kramers = space.kramers()
    rng = np.random.default_rng(1)
    vector = np.ascontiguousarray(rng.standard_normal(space.ndet)
                                  + 1j * rng.standard_normal(space.ndet))
    left = sigma(np.ascontiguousarray(kramers.time_reverse(vector)))
    right = kramers.time_reverse(sigma(vector))
    assert np.max(np.abs(left - right)) < 1e-12 * np.max(np.abs(left))


def test_a_random_hermitian_spinor_hamiltonian_does_not_commute_with_time_reversal():
    """⚠ The control. Without it the check above is compatible with a function that always
    returns zero, and the whole restricted path would be resting on nothing."""
    h, eri = random_spinor_integrals(8, seed=3)
    err_h, err_eri = time_reversal_violation(h, eri)
    assert err_h > 0.1 and err_eri > 0.1
    with pytest.raises(ValueError, match="not time-reversal symmetric"):
        assert_time_reversal(h, eri)


def test_a_kramers_paired_orbital_set_gives_time_reversal_symmetric_integrals(
        paired_active_integrals):
    h, eri, _ = paired_active_integrals
    err_h, err_eri = time_reversal_violation(h, eri)
    assert max(err_h, err_eri) < 1e-13
    assert_time_reversal(h, eri)                       # does not raise


def test_the_projected_integral_set_keeps_hermiticity_and_the_four_fold_relations():
    """The generator above is only a valid test system if the projection preserves what the
    sigma operator asserts on construction — which it does, and which is checked rather than
    argued (``SigmaOperator`` would raise otherwise)."""
    h, eri = time_reversal_symmetric_integrals(8)
    assert np.max(np.abs(h - h.conj().T)) < 1e-14
    assert np.max(np.abs(eri - eri.transpose(2, 3, 0, 1))) < 1e-14
    assert np.max(np.abs(eri - eri.transpose(1, 0, 3, 2).conj())) < 1e-14
    assert max(time_reversal_violation(h, eri)) < 1e-14


# --- K1: the restricted path against the general one -----------------------------------------

@pytest.mark.parametrize("n,k,n_states", [(12, 5, 4), (12, 5, 6), (12, 5, 8), (14, 3, 6),
                                          (12, 7, 4)])
def test_the_restricted_path_reproduces_the_general_one(n, k, n_states):
    space = CASSpace(n, k)
    h, eri = time_reversal_symmetric_integrals(n)
    sigma = SigmaOperator(space, h, eri)
    diagonal = diagonal_energies(space, h, eri)

    general = davidson(sigma, diagonal, n_states, label="general")
    restricted = davidson_kramers(sigma, space.kramers(), diagonal, n_states // 2,
                                  label="restricted")
    assert not general.dense and not restricted.dense, "this is meant to be the iterative path"
    assert np.max(np.abs(general.energies - restricted.energies)) < ENERGY_TOL
    assert restricted.n_apply <= general.n_apply


@pytest.mark.parametrize("n,k,n_states", [(12, 5, 6), (12, 5, 8), (12, 7, 8), (14, 3, 6)])
def test_the_factor_of_two_arrives_above_two_kramers_pairs(n, k, n_states):
    """⚠ The mode's one claim, pinned as a strict inequality on applications of ``H`` — the
    only part of it a correctness test can assert without a stopwatch.

    ⚠ **And it is stated with its boundary, because the boundary is real.** At *two* pairs the
    restricted path adds one expansion direction per iteration where the general path adds
    two, so it needs more iterations and the measured saving collapses to 1.1-1.2x (see
    the CI package's validation record, which carries the CPU seconds). From three pairs upward
    the ratio is 1.84-2.01 across every space measured. A test that asserted the factor of two
    at every root count would be asserting something false, so this one does not.
    """
    space = CASSpace(n, k)
    h, eri = time_reversal_symmetric_integrals(n)
    sigma = SigmaOperator(space, h, eri)
    diagonal = diagonal_energies(space, h, eri)
    general = davidson(sigma, diagonal, n_states, label="general")
    restricted = davidson_kramers(sigma, space.kramers(), diagonal, n_states // 2,
                                  label="restricted")
    assert restricted.n_apply < 0.6 * general.n_apply


@pytest.mark.parametrize("n,k,n_states", [(6, 3, 4), (8, 5, 6)])
def test_the_dense_restricted_path_costs_half_the_applications(n, k, n_states):
    """Below the dense threshold the halving is a certainty rather than a measurement: the
    complete time-reversal-closed basis is one vector per representative determinant."""
    space = CASSpace(n, k)
    h, eri = time_reversal_symmetric_integrals(n)
    sigma = SigmaOperator(space, h, eri)
    diagonal = diagonal_energies(space, h, eri)

    general = davidson(sigma, diagonal, n_states, label="general")
    restricted = davidson_kramers(sigma, space.kramers(), diagonal, n_states // 2,
                                  label="restricted")
    assert general.dense and restricted.dense
    assert restricted.n_apply * 2 == general.n_apply == space.ndet
    assert np.max(np.abs(general.energies - restricted.energies)) < ENERGY_TOL


@pytest.mark.parametrize("n,k,n_states", [(12, 5, 6), (14, 3, 4)])
def test_the_restricted_states_are_orthonormal_and_exactly_paired(n, k, n_states):
    space = CASSpace(n, k)
    h, eri = time_reversal_symmetric_integrals(n)
    sigma = SigmaOperator(space, h, eri)
    result = davidson_kramers(sigma, space.kramers(), diagonal_energies(space, h, eri),
                              n_states // 2, label="restricted")
    overlap = result.vectors.conj() @ result.vectors.T
    assert np.max(np.abs(overlap - np.eye(n_states))) < 1e-11
    # ⚠ Asserted at 0, not at a band: the pair degeneracy here is a copy of one float, not a
    # numerical coincidence. It is a *side effect* of the mode and never the argument for it.
    assert np.array_equal(result.energies[0::2], result.energies[1::2])
    # ... and state 2r+1 really is the time reverse of state 2r, not merely degenerate with it.
    kramers = space.kramers()
    assert np.max(np.abs(kramers.time_reverse(result.vectors[0::2])
                         - result.vectors[1::2])) < 1e-12


@pytest.mark.parametrize("n_pairs", [1, 2, 5])
def test_pair_selection_survives_levels_that_are_more_than_doubly_degenerate(n_pairs):
    """⚠ **The mechanism test for how a Kramers pair is picked out of the subspace.**

    ``eigh`` returns the exactly degenerate members of a pair in an arbitrary basis, so the
    obvious rule — take every other eigenvector — is correct only while no two *pairs* are
    degenerate with each other. A **spin-free** Hamiltonian breaks that immediately: it has
    exact spin multiplets, so an odd-electron spectrum contains blocks of four, and inside a
    4-fold block eigenvectors 0 and 2 are orthogonal while the time-reverse of 0 need not be
    orthogonal to 2. The selection would then return a non-orthonormal set of "states" that
    still had plausible energies.

    So the case is asserted directly, on the same spin-free integrals ``test_casci`` uses for
    the quartets that blind the odd-block state-averaging gate.
    """
    from test_casci import _spin_free_spinor_integrals                 # noqa: PLC0415

    h, eri = _spin_free_spinor_integrals(6, seed=11)
    space = CASSpace(12, 5)
    sigma = SigmaOperator(space, h, eri)
    diagonal = diagonal_energies(space, h, eri)
    general = davidson(sigma, diagonal, 2 * n_pairs, label="general")
    restricted = davidson_kramers(sigma, space.kramers(), diagonal, n_pairs,
                                  label="restricted")
    assert np.max(np.abs(general.energies - restricted.energies)) < 1e-12
    overlap = restricted.vectors.conj() @ restricted.vectors.T
    assert np.max(np.abs(overlap - np.eye(2 * n_pairs))) < 1e-11


def test_the_solver_agrees_with_the_general_path_on_energies_and_rdms(
        paired_active_integrals):
    h, eri, e_core = paired_active_integrals
    results = {}
    for mode in ("general", "restricted"):
        solver = FullCISolver(10, 5, n_states=6, kramers=mode)
        results[mode] = (solver, solver.solve_active(h, eri, e_core=e_core))
    general, restricted = results["general"][1], results["restricted"][1]
    # ⚠ Relative here and absolute in the synthetic tests above, and the difference is not
    # cosmetic: an embedded active-space Hamiltonian carries the inactive Fock, so these
    # eigenvalues sit near -390 Eh where one ulp is already 5e-14 and an absolute 1e-12 would
    # be a test of float64's exponent rather than of this code.
    assert np.allclose(general.energies, restricted.energies, rtol=1e-13, atol=ENERGY_TOL)
    assert np.max(np.abs(general.gamma - restricted.gamma)) < RDM_TOL
    assert np.max(np.abs(general.gamma2 - restricted.gamma2)) < RDM_TOL
    assert np.allclose(general.weights, restricted.weights)
    assert results["restricted"][0].n_apply < 0.6 * results["general"][0].n_apply


def test_a_restricted_solve_averages_whole_pairs_so_the_gate_has_nothing_to_refuse(
        paired_active_integrals):
    """⚠ Asserted rather than assumed. The state-averaging gate stays in charge — the mode
    does not reimplement it, it merely cannot present it with a split block."""
    from kuiva.rdm.rdm import degenerate_blocks                        # noqa: PLC0415

    h, eri, e_core = paired_active_integrals
    solver = FullCISolver(10, 5, n_states=6, kramers="restricted")
    result = solver.solve_active(h, eri, e_core=e_core)
    for start, stop in degenerate_blocks(result.energies):
        assert (stop - start) % 2 == 0
    assert np.allclose(result.weights, 1.0 / 6.0)


def test_the_transition_densities_agree_through_a_phase_invariant_reduction(
        paired_active_integrals):
    """⚠ Element-by-element comparison is meaningless here and would be a *wrong* test.

    Inside a degenerate manifold the individual states are defined only up to a unitary
    mixing, and the two symmetry modes make different, equally valid choices. What is
    comparable is the block invariant ``Tr_block(P_i P_j)`` — the same reduction cross-code
    validation of the property dump is restricted to.
    """
    h, eri, e_core = paired_active_integrals
    rng = np.random.default_rng(12)
    n_act = 10
    operators = rng.standard_normal((3, n_act, n_act)) + 1j * rng.standard_normal(
        (3, n_act, n_act))
    operators = 0.5 * (operators + operators.conj().transpose(0, 2, 1))

    invariants = {}
    for mode in ("general", "restricted"):
        solver = FullCISolver(n_act, 5, n_states=6, kramers=mode)
        solver.solve_active(h, eri, e_core=e_core)
        gamma_ij = solver.transition_densities()               # (n_states, n_states, n, n)
        matrices = np.einsum("kpq,ijpq->kij", operators, gamma_ij)
        invariants[mode] = np.stack([block_moment_tensor(matrices, start, 2)
                                     for start in (0, 2, 4)])
    assert np.max(np.abs(invariants["general"] - invariants["restricted"])) < 1e-8


# --- The refusals ------------------------------------------------------------------------------

def test_an_even_electron_count_is_refused_as_a_different_theorem():
    with pytest.raises(ValueError, match="odd-electron theorem"):
        FullCISolver(10, 4, n_states=2, kramers="restricted")


def test_an_odd_state_count_is_refused():
    with pytest.raises(ValueError, match="whole Kramers pairs"):
        FullCISolver(10, 5, n_states=3, kramers="restricted")


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="kramers must be one of"):
        FullCISolver(10, 5, n_states=2, kramers="quaternion")


def test_an_odd_spinor_count_is_refused():
    with pytest.raises(ValueError, match="even number of spinors"):
        FullCISolver(9, 5, n_states=2, kramers="restricted")


def test_integrals_that_break_the_pairing_are_refused_at_the_solve():
    """⚠ The check that stands between a broken Kramers pairing and a converged, exactly
    degenerate, wrong spectrum. It runs at **every** solve because a CASSCF rotates the
    orbitals at every macro-iteration."""
    h, eri = random_spinor_integrals(8, seed=3)
    solver = FullCISolver(8, 3, n_states=2, kramers="restricted")
    with pytest.raises(ValueError, match="not time-reversal symmetric"):
        solver.solve_active(h, eri)


def test_the_even_parity_map_is_refused_by_the_restricted_eigensolver():
    """Defence in depth: the solver refuses an even count at construction, and the eigensolver
    refuses the map itself, so neither can be reached through the other."""
    space = CASSpace(8, 4)
    h, eri = time_reversal_symmetric_integrals(8)
    sigma = SigmaOperator(space, h, eri)
    with pytest.raises(ValueError, match="odd-electron theorem"):
        davidson_kramers(sigma, space.kramers(), diagonal_energies(space, h, eri), 2)


# --- K2: the mode through the CASSCF driver and the machinery downstream ----------------------

def test_the_symmetry_mode_is_part_of_the_space_key():
    """⚠ So that a restart or a mid-run switch is a **chart change**: the optimizer's curvature
    memory is cleared rather than transported across two different solvers."""
    general = FullCISolver(10, 5, n_states=2, kramers="general")
    restricted = FullCISolver(10, 5, n_states=2, kramers="restricted")
    assert general.space_key() != restricted.space_key()
    assert "restricted" in restricted.space_key()


def test_selecting_the_mode_announces_itself(kuiva_caplog):
    """A non-default symmetry mode says so at the point of selection, like every other
    non-default Hamiltonian choice in the program."""
    with kuiva_caplog.at_level("INFO"):
        FullCISolver(10, 5, n_states=2, kramers="restricted")
    assert any("Kramers-restricted" in record.getMessage()
               for record in kuiva_caplog.records)


def test_a_casscf_converges_to_the_same_answer_in_both_modes():
    """The whole chain, both ways, on orbitals that start Kramers paired.

    ⚠ It also asserts the thing the mode's precondition rests on: the general-complex
    optimizer is *free* to leave the Kramers-preserving subgroup, and this says it does not —
    the converged orbitals are still paired to roundoff, which is why the restricted mode's
    per-solve check never fires along a physical trajectory.
    """
    factors, h_ao, coeff, spaces = kramers_paired_system(nao=6, n_inactive=2, n_active=6)
    outcomes = {}
    for mode in ("general", "restricted"):
        solver = FullCISolver(6, 3, n_states=4, kramers=mode)
        outcomes[mode] = casscf(factors, h_ao, coeff, spaces, 3, n_states=4, solver=solver,
                                report=False, boundary_check=0, max_iter=60)
    general, restricted = outcomes["general"], outcomes["restricted"]
    assert general.converged and restricted.converged
    assert abs(general.energy - restricted.energy) < 1e-10
    assert np.max(np.abs(general.state_energies - restricted.state_energies)) < 1e-10

    converged = CASIntegrals.build(factors, h_ao, general.coeff, spaces, e_nuc=0.0)
    err = time_reversal_violation(np.ascontiguousarray(converged.h_active_effective()),
                                  converged.active_eri())
    assert max(err) < 1e-11


def test_the_whole_chain_downstream_of_the_ci_is_indifferent_to_the_mode():
    """⚠ **The claim that the mode is an option and not a second pipeline**, end to end.

    A real molecule through the class API: front end, CASSCF and SC-NEVPT2, once per mode.
    Everything after the CI consumes pair-expanded states in the general convention, so if any
    of it had learned something about the symmetry mode this is where it would show — and the
    perturbation stage is the sharpest of them, because it rebuilds ``CASSpace`` on the ``N±1``
    and ``N±2`` sectors, where ``T²`` alternates sign and the restricted construction does not
    exist at all.

    ``screening="none"`` because this asserts nothing about the spin-orbit treatment and a
    four-component atomic solve per element is pure cost here — which is also what keeps a
    whole-pipeline Tier-1 check inside the default laptop suite at ~1.5 s.
    """
    import kuiva                                                       # noqa: PLC0415

    molecule = kuiva.Molecule(atoms=[("B", (0.0, 0.0, 0.0))], basis="x2c-SVPall-2c",
                              charge=0, spin=1)
    scf = kuiva.ScalarSCF(molecule, memory_gb=6.0, screening="none").run()
    reference = kuiva.Reference(scf).run()
    outcomes = {}
    for mode in ("general", "restricted"):
        cas = kuiva.CASSCF(reference, character=("B", "p"), n_active=6, n_active_elec=1,
                           n_states=6, solver_options=dict(kramers=mode),
                           max_iter=40).run()
        outcomes[mode] = (cas, kuiva.NEVPT2(cas).run())
    general, restricted = outcomes["general"], outcomes["restricted"]
    assert abs(general[0].energy - restricted[0].energy) < 1e-11
    assert np.max(np.abs(general[0].ci.excitation_energies_cm()
                         - restricted[0].ci.excitation_energies_cm())) < 1e-6
    assert np.max(np.abs(np.asarray(general[1].e2)
                         - np.asarray(restricted[1].e2))) < 1e-12


def test_the_boundary_diagnostic_runs_in_the_restricted_mode(paired_active_integrals):
    """The extra roots the boundary check needs are a margin, not a pair count, so an odd
    request is rounded up to a whole pair and truncated back. Both modes must return the same
    spectrum of the same length."""
    h, eri, e_core = paired_active_integrals
    spectra = {}
    for mode in ("general", "restricted"):
        solver = FullCISolver(10, 5, n_states=4, kramers=mode)
        solver.solve_active(h, eri, e_core=e_core)
        spectra[mode] = solver.spectrum(h, eri, 11, e_core=e_core)
    assert spectra["general"].shape == (11,)
    assert spectra["restricted"].shape == (11,)
    assert np.max(np.abs(spectra["general"] - spectra["restricted"])) < 1e-10
