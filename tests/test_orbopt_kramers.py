"""Tier-0 tests for the Kramers constraint on the CASSCF rotation.

The defect this closes was a *drift*, not a blunder: a general complex rotation does not hold
the orbital spaces closed under time reversal, and the optimizer amplifies the roundoff
asymmetry it injects there until an odd-electron spectrum visibly splits and the
state-averaging gate refuses. Two things follow for what is tested here.

**Test the mechanism, not only the observable.** The observable — a split Kramers pair — is
what made the defect visible, but it is a downstream consequence and only an odd electron
count has one at all. The mechanism is that ``exp(kappa)`` maps Kramers pairs to Kramers pairs
exactly when ``kappa`` obeys ``kappa_pq = t_p t_q conj(kappa[pbar, qbar])``, so that relation,
its projection and the invariance of the pairing under a rotation built from it are what carry
the weight below.

**The synthetic system is deliberately time-reversal symmetric**, which the fixture in
``test_mcscf.py`` is not: a random Hermitian two-component Hamiltonian is time-odd, its
eigenvectors are not Kramers pairs, and neither the defect nor the constraint exists there.
That is also why the two suites do not share a fixture — and why ``"auto"`` leaves the
``test_mcscf.py`` system unconstrained, which is asserted here too.
"""
import itertools

import numpy as np
import pytest

from kuiva.ci.strings import Determinants, hamiltonian_matrix, rdm12
from kuiva.integrals.transform import ThreeIndexAO
from kuiva.mcscf.orbopt import (CASIntegrals, OrbitalHessian, OrbitalOptimizer, OrbitalSpaces,
                                _pack, _unpack, kramers_mirror_mean, kramers_odd_project,
                                kramers_parameter_map, kramers_project,
                                kramers_release_rotation, kramers_rotation_note,
                                lowest_projected_curvature, measure_time_odd_curvature,
                                optimize_orbitals, resolve_kramers_rotation,
                                unitary_from_antihermitian)
from kuiva.spinor.expand import (expand_scalar_mos, kramers_pairing_defect,
                                 time_reversal_closure_defect, time_reversal_even_part,
                                 time_reversal_odd_norm, two_component_operator)


@pytest.fixture(scope="module")
def kramers_system():
    """A small **time-reversal-symmetric** two-component system with a Kramers-paired guess.

    Spin-free part real symmetric, spin-orbit factors real antisymmetric — the structure
    ``kuiva.spinor.expand.two_component_operator`` documents, and the one every real X2C
    Hamiltonian has. The three-index factors are over scalar AO pairs, so the two-electron
    part is spin free and time even by construction. Three active electrons: an odd count, so
    Kramers' theorem gives the spectrum an exact degeneracy that any drift breaks.
    """
    rng = np.random.default_rng(11)
    nao = 6
    npair = nao * (nao + 1) // 2
    factors = ThreeIndexAO(l_packed=0.3 * rng.standard_normal((2 * nao, npair)), nao=nao,
                           origin="cholesky")
    a = rng.standard_normal((nao, nao))
    a_sf = a + a.T
    w = rng.standard_normal((3, nao, nao))
    w = 0.2 * (w - np.transpose(w, (0, 2, 1)))
    h_ao = two_component_operator(a_sf, w)
    _, phi = np.linalg.eigh(a_sf)                       # real scalar MOs: the scalar guess
    coeff = expand_scalar_mos(phi).c
    spaces = OrbitalSpaces.from_counts(n_inactive=2, n_active=4, n_orb=2 * nao)
    return factors, h_ao, np.ascontiguousarray(coeff), spaces, 3


def exact_ci(spaces, nelec):
    """A state-averaged exact CI over the whole degenerate ground manifold."""
    def solver(ints):
        occ = list(itertools.combinations(range(spaces.n_active), nelec))
        dets = Determinants.from_occupations(occ, spaces.n_active)
        mat = hamiltonian_matrix(dets, ints.h_active_effective(),
                                 ints.active_eri()).toarray()
        w, v = np.linalg.eigh(mat)
        g1, g2 = rdm12(dets, v[:, 0])
        g1b, g2b = rdm12(dets, v[:, 1])                 # the Kramers partner: equal weights
        return 0.5 * (w[0] + w[1]) + ints.e_core, 0.5 * (g1 + g1b), 0.5 * (g2 + g2b)
    return solver


def ci_levels(factors, h_ao, coeff, spaces, nelec):
    ints = CASIntegrals.build(factors, h_ao, coeff, spaces, e_nuc=0.0)
    occ = list(itertools.combinations(range(spaces.n_active), nelec))
    dets = Determinants.from_occupations(occ, spaces.n_active)
    mat = hamiltonian_matrix(dets, ints.h_active_effective(), ints.active_eri()).toarray()
    return np.linalg.eigvalsh(mat)


# --- the projector ------------------------------------------------------------------------
def test_packed_projection_is_the_matrix_projection():
    """The packed form and the matrix form are the *same* projection, bitwise.

    Two representations of one convention is the failure mode the single-definition rule
    exists to prevent, so the packed map is checked against ``time_reversal_even_part``
    rather than against its own algebra —
    including with active-active rotations on, where a parameter's time-reversed partner is
    stored as its anti-Hermitian mirror and the naive gather would be wrong by a sign.
    """
    rng = np.random.default_rng(5)
    n = 12
    spaces = OrbitalSpaces.from_counts(4, 4, n)
    for active_active in (False, True):
        rows, cols = spaces.rotation_pairs(active_active=active_active)
        kmap = kramers_parameter_map(rows, cols, n)
        assert kmap is not None
        v = rng.standard_normal(rows.size) + 1j * rng.standard_normal(rows.size)
        packed = kramers_project(v, kmap)
        matrix = _pack(time_reversal_even_part(_unpack(v, rows, cols, n)), rows, cols)
        assert np.array_equal(packed, matrix)
        # A projection: idempotent to the bit, and what it produces is exactly even.
        assert np.array_equal(kramers_project(packed, kmap), packed)
        assert time_reversal_odd_norm(_unpack(packed, rows, cols, n)) == 0.0
        # Anti-hermiticity survives it, so the projected generator still exponentiates to a
        # unitary — the projection is inside the parametrization, not a replacement for it.
        kappa = _unpack(packed, rows, cols, n)
        assert np.max(np.abs(kappa + kappa.conj().T)) == 0.0


def test_the_projection_is_orthogonal_for_the_optimizers_inner_product():
    """``Re <a, b>`` is the inner product every step length, curvature pair and convergence
    test is measured in, and the projection has to be orthogonal *for that one* — otherwise
    the constrained problem is not a restriction of the unconstrained one and a projected
    descent direction need not descend."""
    rng = np.random.default_rng(7)
    n = 10
    spaces = OrbitalSpaces.from_counts(2, 4, n)
    rows, cols = spaces.rotation_pairs()
    kmap = kramers_parameter_map(rows, cols, n)
    a = rng.standard_normal(rows.size) + 1j * rng.standard_normal(rows.size)
    b = rng.standard_normal(rows.size) + 1j * rng.standard_normal(rows.size)
    pa, pb = kramers_project(a, kmap), kramers_project(b, kmap)
    assert np.real(np.vdot(pa, b)) == pytest.approx(np.real(np.vdot(pa, pb)), abs=1e-12)
    assert np.real(np.vdot(a, pb)) == pytest.approx(np.real(np.vdot(pa, pb)), abs=1e-12)


def test_mirror_mean_symmetrizes_the_preconditioner():
    rng = np.random.default_rng(9)
    n = 8
    spaces = OrbitalSpaces.from_counts(2, 2, n)
    rows, cols = spaces.rotation_pairs()
    kmap = kramers_parameter_map(rows, cols, n)
    h = np.abs(rng.standard_normal(rows.size)) + 0.5
    mean = kramers_mirror_mean(h, kmap)
    assert np.allclose(mean, mean[kmap[0]], atol=0.0)


# --- the mechanism ------------------------------------------------------------------------
def test_an_even_generator_preserves_kramers_pairing_and_a_general_one_destroys_it():
    """**The mechanism the whole fix rests on.** ``exp`` of a time-reversal-even generator
    maps Kramers pairs to Kramers pairs; a general anti-Hermitian generator of the same size
    does not, and the failure is O(1) rather than small."""
    rng = np.random.default_rng(13)
    nao, n = 5, 10
    phi = np.linalg.qr(rng.standard_normal((nao, nao)))[0]
    c = expand_scalar_mos(phi).c
    assert kramers_pairing_defect(c) == 0.0
    a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    general = 0.05 * (a - a.conj().T)
    even = time_reversal_even_part(general)
    assert kramers_pairing_defect(c @ unitary_from_antihermitian(even)) < 1e-14
    assert kramers_pairing_defect(c @ unitary_from_antihermitian(general)) > 1e-3


def test_the_exact_gradient_of_a_symmetric_problem_is_already_even(kramers_system):
    """Why projecting is safe rather than a bias: with a time-reversal-symmetric Hamiltonian
    and a time-even ensemble, ``E(kappa) = E(Theta kappa)`` identically, so the odd part of
    the gradient is roundoff and nothing else. The constraint discards noise, not signal."""
    factors, h_ao, coeff, spaces, nelec = kramers_system
    ints = CASIntegrals.build(factors, h_ao, coeff, spaces, e_nuc=0.0)
    energy, gamma, gamma2 = exact_ci(spaces, nelec)(ints)
    opt = OrbitalOptimizer(spaces, mode="quasi-newton")
    grad, _hdiag = opt.gradient(ints, gamma, gamma2, factors, coeff)
    assert time_reversal_odd_norm(_unpack(grad, opt.rows, opt.cols, spaces.n_orb)) < 1e-13
    assert np.linalg.norm(grad) > 1e-3          # ...and it is not simply a zero gradient


def test_parameter_map_refuses_a_space_that_splits_a_kramers_pair():
    """The structural precondition, and it fails loudly. A space holding half a pair has no
    pair-closed parameter list, so there is no constraint to impose — ``"auto"`` leaves the
    rotation alone and an explicit request refuses."""
    n = 10
    split = OrbitalSpaces(inactive=np.arange(3), active=np.arange(3, 7),
                          virtual=np.arange(7, n), n_orb=n)
    rows, cols = split.rotation_pairs()
    assert kramers_parameter_map(rows, cols, n) is None
    with pytest.raises(ValueError, match="closed under the Kramers index swap"):
        OrbitalOptimizer(split, kramers=True)


def test_auto_measures_the_orbitals_rather_than_trusting_a_flag(kramers_system):
    factors, h_ao, coeff, spaces, nelec = kramers_system
    on, pairing = resolve_kramers_rotation("auto", coeff, spaces)
    assert on and pairing == 0.0
    # An unpaired set — what an unrestricted reference gives, and what an unconstrained
    # CASSCF converges to — is left alone by "auto" and refused by an explicit request.
    unpaired = np.array(coeff, copy=True)
    unpaired[:, 3] *= np.exp(0.3j)
    assert resolve_kramers_rotation("auto", unpaired, spaces)[0] is False
    with pytest.raises(ValueError, match="needs Kramers-paired orbitals"):
        resolve_kramers_rotation(True, unpaired, spaces)
    with pytest.raises(ValueError, match="must be True, False or 'auto'"):
        resolve_kramers_rotation("yes", coeff, spaces)


def test_auto_constrains_a_paired_set_at_either_parity(kramers_system):
    """⚠ **The electron count no longer decides the constraint** (it did in v0.36.0).

    A time-reversal-broken solution is a legitimate variational answer at an even count, so
    the constraint could not simply be imposed there — but declining it left every
    even-electron run exposed to the drift, and parity is evidence about neither. What
    decides now is a measurement at the converged point (:func:`measure_time_odd_curvature`),
    so ``"auto"`` constrains whatever is Kramers paired and the release is what parity still
    governs. The two ways of coming back unconstrained stay different statements and are
    still reported differently.
    """
    _factors, _h, coeff, spaces, _nelec = kramers_system
    for nelec in (3, 4):                                # parity is not consulted at all
        assert resolve_kramers_rotation("auto", coeff, spaces)[0] is True
    assert resolve_kramers_rotation(True, coeff, spaces)[0] is True
    assert "spaces stay time-reversal closed" in kramers_rotation_note(
        *resolve_kramers_rotation("auto", coeff, spaces))
    unpaired = np.array(coeff, copy=True)
    unpaired[:, 3] *= np.exp(0.3j)
    assert "not Kramers paired" in kramers_rotation_note(
        *resolve_kramers_rotation("auto", unpaired, spaces))
    assert "by request" in kramers_rotation_note(
        *resolve_kramers_rotation(False, coeff, spaces))


# --- end to end ---------------------------------------------------------------------------
def test_a_constrained_optimization_stays_in_the_closed_manifold(kramers_system):
    """The observable, at the end of the mechanism: over a whole optimization the orbitals
    stay Kramers paired at roundoff, every orbital space stays time-reversal closed, and the
    odd-electron CI spectrum stays exactly doubly degenerate — while the same run with the
    constraint off leaves all three, which is the drift this exists to stop."""
    factors, h_ao, coeff, spaces, nelec = kramers_system
    solver = exact_ci(spaces, nelec)
    blocks = [spaces.inactive, spaces.active, spaces.virtual]

    results = {}
    for constrained in (True, False):
        res = optimize_orbitals(factors, h_ao, coeff, spaces, solver, max_iter=60,
                                conv_grad=1e-6, report=False,
                                kramers_rotation=constrained)
        levels = ci_levels(factors, h_ao, res.coeff, spaces, nelec)
        results[constrained] = (kramers_pairing_defect(res.coeff),
                                time_reversal_closure_defect(res.coeff, blocks),
                                float(np.max(levels[1::2] - levels[0::2])), res)

    pairing, closure, split, res = results[True]
    assert pairing < 1e-12, "pairing defect {:.2e}".format(pairing)
    assert closure < 1e-12, "closure defect {:.2e}".format(closure)
    assert split < 1e-10, "Kramers splitting {:.2e} Eh".format(split)
    assert res.converged and res.energy < res.history[0]
    # ⚠ **No contrast is asserted against the unconstrained run here, and that is a
    # measurement rather than an omission**: this system converges in ~15 macro-iterations,
    # which is far too few for the amplification to compound, and the unconstrained run
    # leaves the manifold by 8e-15 — the same roundoff floor. The drift needs a stiff
    # problem and dozens of steps (UF3 took thirteen solves to reach 5e-7). The contrast a
    # test *can* hold deterministically is the one below.
    assert results[False][0] < 1e-12


def test_an_unconstrained_run_is_bitwise_what_it_was(kramers_system):
    """``kramers_rotation=False`` and an ``"auto"`` that resolves to off must be the *same*
    trajectory, bitwise: the constraint may not change a calculation it does not apply to."""
    factors, h_ao, coeff, spaces, nelec = kramers_system
    unpaired = np.array(coeff, copy=True)
    unpaired[:, 3] *= np.exp(0.3j)                  # "auto" declines on this set
    solver = exact_ci(spaces, nelec)
    a = optimize_orbitals(factors, h_ao, unpaired, spaces, solver, max_iter=12,
                          conv_grad=1e-6, report=False, kramers_rotation="auto")
    b = optimize_orbitals(factors, h_ao, unpaired, spaces, solver, max_iter=12,
                          conv_grad=1e-6, report=False, kramers_rotation=False)
    assert a.energy == b.energy
    assert np.array_equal(a.coeff, b.coeff)
    assert a.n_iterations == b.n_iterations


def test_the_constraint_holds_the_manifold_where_an_unconstrained_rotation_leaves_it(
        kramers_system):
    """The rescue, made deterministic.

    The real drift is roundoff in the time-odd rotation directions, amplified step over step
    by curvature the energy barely resists; it needs a stiff problem and dozens of steps to
    become visible, which no laptop-fast test has. What *is* reproducible in a few steps is
    the same mechanism driven at an amplitude one can see: a solver returning a slightly
    time-odd 1-RDM — which is what a truncated or adaptive solver genuinely returns, and what
    roundoff imitates — puts a real component in the odd directions of the gradient, and the
    unconstrained rotation follows it straight out of the Kramers manifold (0.3, i.e. gone)
    while the constrained one stays at roundoff.

    ⚠ It is a **model of the drift, not the drift**: the amplitude is injected rather than
    accumulated. What it tests is what the constraint does with an odd gradient component,
    which is the step the fix acts on.
    """
    factors, h_ao, coeff, spaces, nelec = kramers_system
    base = exact_ci(spaces, nelec)
    rng = np.random.default_rng(17)
    n_act = spaces.n_active
    skew = rng.standard_normal((n_act, n_act)) + 1j * rng.standard_normal((n_act, n_act))
    skew = 1e-3 * (skew + skew.conj().T)             # Hermitian, and time-reversal odd

    def skewed(ints):
        energy, gamma, gamma2 = base(ints)
        return energy, gamma + skew, gamma2

    defects = {}
    for constrained in (True, False):
        res = optimize_orbitals(factors, h_ao, coeff, spaces, skewed, max_iter=15,
                                conv_grad=1e-6, report=False, mode="second-order",
                                kramers_rotation=constrained)
        defects[constrained] = kramers_pairing_defect(res.coeff)
    assert defects[False] > 1e-4, "the model no longer drives the drift it is here to model"
    assert defects[True] < 1e-12, "pairing defect {:.2e}".format(defects[True])


# --- is the constrained solution a minimum, or a saddle in the directions it forbids? ------
def test_the_odd_projection_is_the_complement_of_the_even_one():
    """``P_even + P_odd = 1``, each idempotent, and the two orthogonal in the optimizer's own
    real inner product — which is what makes the stability question a *decomposition* of the
    unconstrained problem rather than a second one. Without it, "the curvature the constraint
    forbids" would not be the complement of "the curvature it optimizes in"."""
    rng = np.random.default_rng(3)
    n = 12
    spaces = OrbitalSpaces.from_counts(4, 4, n)
    for active_active in (False, True):
        rows, cols = spaces.rotation_pairs(active_active=active_active)
        kmap = kramers_parameter_map(rows, cols, n)
        v = rng.standard_normal(rows.size) + 1j * rng.standard_normal(rows.size)
        even, odd = kramers_project(v, kmap), kramers_odd_project(v, kmap)
        assert np.allclose(even + odd, v, rtol=0, atol=1e-15)
        assert np.allclose(kramers_odd_project(odd, kmap), odd, rtol=0, atol=1e-15)
        assert np.allclose(kramers_project(odd, kmap), 0.0, rtol=0, atol=1e-15)
        assert abs(float(np.real(np.vdot(even, odd)))) < 1e-13
        # And what it produces is exactly time-odd: its even part vanishes as a matrix too.
        kappa = _unpack(odd, rows, cols, n)
        assert np.max(np.abs(time_reversal_even_part(kappa))) < 1e-15


def test_the_projected_davidson_finds_the_dense_lowest_curvature(kramers_system):
    """The eigensolver against the same operator built densely on the odd subspace.

    A stability verdict is a **sign**, and a Davidson that missed the lowest eigenvector would
    report the wrong one while converging happily — so the check is against an exact
    diagonalization of the restricted operator, not against the solver's own residual.
    """
    factors, h_ao, coeff, spaces, nelec = kramers_system
    ints = CASIntegrals.build(factors, h_ao, coeff, spaces, e_nuc=0.0)
    _energy, gamma, gamma2 = exact_ci(spaces, nelec)(ints)
    opt = OrbitalOptimizer(spaces, kramers=True)
    hess = OrbitalHessian(ints, factors, h_ao, coeff, gamma, gamma2, opt.rows, opt.cols)
    project = lambda v: kramers_odd_project(v, opt.kmap)          # noqa: E731

    # A real orthonormal basis of the odd subspace: the projector is real-linear, so a
    # parameter's real and imaginary directions are two independent vectors.
    basis = []
    for k in range(opt.n_parameters):
        for value in (1.0 + 0.0j, 0.0 + 1.0j):
            seed = np.zeros(opt.n_parameters, dtype=np.complex128)
            seed[k] = value
            v = project(seed)
            for b in basis:
                v = v - float(np.real(np.vdot(b, v))) * b
            nrm = float(np.sqrt(np.real(np.vdot(v, v))))
            if nrm > 1e-8:
                basis.append(v / nrm)
    applied = [project(hess.matvec(b)) for b in basis]
    dense = np.array([[float(np.real(np.vdot(a, b))) for b in applied] for a in basis])
    exact = float(np.linalg.eigvalsh(0.5 * (dense + dense.T))[0])

    lam, vec, _n, converged = lowest_projected_curvature(
        hess.matvec, hess.exact_diagonal(), project)
    assert converged
    assert abs(lam - exact) < 1e-6 * max(abs(exact), 1.0), "{} vs {}".format(lam, exact)
    # The eigenvector is in the subspace, normalized, and its sign is fixed rather than
    # left to roundoff — two runs of one script may not step off a saddle in two directions.
    assert np.allclose(project(vec), vec, atol=1e-12)
    assert abs(float(np.sqrt(np.real(np.vdot(vec, vec)))) - 1.0) < 1e-10
    again = lowest_projected_curvature(hess.matvec, hess.exact_diagonal(), project)[1]
    assert np.array_equal(vec, again)


def test_the_release_step_breaks_the_symmetry_it_is_meant_to_break(kramers_system):
    """The nudge is what makes a release a release.

    At the constrained solution the gradient vanishes in **every** direction, so dropping the
    constraint alone would leave the run stationary and it would converge again on the spot.
    The step off the saddle is therefore part of the mechanism: it is unitary, it is scaled to
    the stated angle, and the orbitals it produces are measurably *not* Kramers paired.
    """
    _factors, _h, coeff, spaces, _nelec = kramers_system
    opt = OrbitalOptimizer(spaces, kramers=True)
    rng = np.random.default_rng(5)
    v = rng.standard_normal(opt.n_parameters) + 1j * rng.standard_normal(opt.n_parameters)
    direction = kramers_odd_project(v, opt.kmap)
    u = kramers_release_rotation(direction, opt.rows, opt.cols, spaces.n_orb,
                                 max_rotation=0.1)
    assert np.max(np.abs(u.conj().T @ u - np.eye(spaces.n_orb))) < 1e-13
    moved = np.ascontiguousarray(coeff @ u)
    assert kramers_pairing_defect(coeff) < 1e-14 < kramers_pairing_defect(moved)
    # Scaled, not normalized: the angle is the knob, and it is the one that was asked for.
    smaller = kramers_release_rotation(direction, opt.rows, opt.cols, spaces.n_orb,
                                       max_rotation=0.01)
    assert (kramers_pairing_defect(np.ascontiguousarray(coeff @ smaller))
            < kramers_pairing_defect(moved))


def test_a_stable_solution_keeps_its_constraint_and_says_what_was_measured(kramers_system):
    """A healthy even-electron run: positive curvature, no release, and the measurement
    changes no number — it is a report about the answer, not part of finding it."""
    factors, h_ao, coeff, spaces, _nelec = kramers_system
    solver = exact_ci(spaces, 2)
    common = dict(max_iter=60, conv_grad=1e-6, report=False, mode="second-order",
                  n_active_elec=2)
    tested = optimize_orbitals(factors, h_ao, coeff, spaces, solver, **common)
    assert tested.converged and tested.kramers_released is False
    assert tested.time_odd_curvature is not None and tested.time_odd_curvature > 0.0
    quiet = optimize_orbitals(factors, h_ao, coeff, spaces, solver,
                              kramers_stability=False, **common)
    assert quiet.time_odd_curvature is None                 # not measured is not "stable"
    assert tested.energy == quiet.energy
    assert np.array_equal(tested.coeff, quiet.coeff)
    assert tested.n_iterations == quiet.n_iterations


def test_the_measurement_runs_only_where_a_release_could_be_acted_on(kramers_system):
    """⚠ The default measures where it could *do* something and nowhere else, because it
    costs Hessian-vector products. An odd count has no Kramers-degenerate broken solution to
    release to, and an unstated count is not a guess at one — both are left unmeasured by
    ``"auto"`` and measured by an explicit ``True``, which is the diagnostic setting."""
    factors, h_ao, coeff, spaces, _nelec = kramers_system
    common = dict(max_iter=60, conv_grad=1e-6, report=False, mode="second-order")
    odd = optimize_orbitals(factors, h_ao, coeff, spaces, exact_ci(spaces, 3),
                            n_active_elec=3, **common)
    assert odd.converged and odd.time_odd_curvature is None
    told = optimize_orbitals(factors, h_ao, coeff, spaces, exact_ci(spaces, 3),
                             n_active_elec=3, kramers_stability=True, **common)
    assert told.time_odd_curvature is not None and told.kramers_released is False
    unstated = optimize_orbitals(factors, h_ao, coeff, spaces, exact_ci(spaces, 2), **common)
    assert unstated.time_odd_curvature is None
    # An unconstrained run has nothing to measure: the whole space was already searched.
    free = optimize_orbitals(factors, h_ao, coeff, spaces, exact_ci(spaces, 2),
                             kramers_rotation=False, kramers_stability=True,
                             n_active_elec=2, **common)
    assert free.time_odd_curvature is None
    with pytest.raises(ValueError, match="the time-odd curvature is defined against"):
        measure_time_odd_curvature(OrbitalOptimizer(spaces), None, factors, h_ao, coeff,
                                   None, None)
    with pytest.raises(ValueError, match="kramers_stability must be True, False or 'auto'"):
        optimize_orbitals(factors, h_ao, coeff, spaces, exact_ci(spaces, 2),
                          kramers_stability="yes", **common)


@pytest.mark.slow
def test_n2_releases_the_constraint_and_recovers_the_broken_solution():
    """The characterized instability, end to end: N2 CAS(6,8), the one case where a
    time-reversal-broken solution is real rather than drift.

    ⚠ **No synthetic reproduces this**, which is why the test is here and slow: a
    symmetry-breaking instability needs a genuinely poor active space — this one's inactive
    space forces an orbital the SCF leaves empty, so the CAS cannot hold the reference
    determinant — and all sixty configurations of the fast suite's model system, misordered
    spaces included, come back stable at both parities.

    What is asserted is the pair of statements the mechanism exists to separate: constrained,
    the run stops at the symmetric stationary point; with the stability test on, it measures
    a negative time-odd curvature there, follows it, and arrives at the **same** solution an
    unconstrained run from the scalar guess reaches. ⚠ The two are compared as *converged*
    energies, never as trajectories: the unconstrained route is a ridge here (two identical
    invocations were measured 0.41 Eh apart at other state counts).
    """
    pytest.importorskip("pyscf")
    import kuiva

    molecule = kuiva.Molecule(atoms=[("N", (0.0, 0.0, 0.0)), ("N", (0.0, 0.0, 1.098))],
                              basis="x2c-SVPall-2c")
    reference = kuiva.Reference(
        kuiva.ScalarSCF(molecule, memory_gb=6.0, screening="none").run()).run()
    common = dict(character=([0, 1], "p"), n_active=8, n_active_elec=6,
                  mode="second-order", max_iter=60, conv_grad=1e-4, report=False)
    released = kuiva.CASSCF(reference, **common).run()
    constrained = kuiva.CASSCF(reference, kramers_stability=False, **common).run()
    free = kuiva.CASSCF(reference, kramers_rotation=False, **common).run()

    assert constrained.converged and released.converged and free.converged
    # The constraint alone stops at the symmetric solution, which is 0.64 Eh too high.
    assert abs(constrained.energy - (-108.3062)) < 1e-3
    assert constrained.orbital.time_odd_curvature is None
    # The measurement finds the saddle it is there to find, and follows it to the answer.
    assert released.orbital.kramers_released is True
    assert released.orbital.time_odd_curvature < -0.1
    assert abs(released.energy - free.energy) < 1e-6
    assert released.energy < constrained.energy - 0.6
