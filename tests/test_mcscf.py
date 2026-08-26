"""Tier-0/1 tests for the shared orbital optimizer and the cheap CI.

The orbital gradient is the load-bearing quantity: the CASSCF fixed point is *defined* by it
vanishing, so an error there does not produce a visibly wrong answer, it produces a
confidently converged wrong answer. It is therefore checked three ways — against finite
differences of the energy, against the general generalized-Fock formula built from explicit
total density matrices, and through invariances that must hold exactly (redundant rotations
change nothing; a converged CI has no active-active gradient).

The cheap CI is checked against the one thing it *must* get right (a full space reproduces
FCI exactly) and against what it is actually for: natural occupations and entanglement, which
must converge with the determinant budget much faster than the energy does — that is the
premise the whole method rests on.
"""
import itertools

import numpy as np
import pytest

from kuiva.ci.strings import Determinants, hamiltonian_matrix, rdm12
from kuiva.integrals.transform import (ThreeIndexAO, assemble_4c, coulomb_exchange,
                                       transform_1e, transform_3c)
from kuiva.mcscf.orbopt import (CASIntegrals, OrbitalHessian, OrbitalOptimizer, OrbitalSpaces,
                                _active_fock, _pack, _unpack, augmented_hessian_step,
                                cas_energy, diagonal_hessian, generalized_fock,
                                optimize_orbitals, orbital_gradient,
                                unitary_from_antihermitian)
from kuiva.mcscf.preopt import cheap_ci, reference_determinants
from kuiva.rdm.entropy import (fiedler_order, mutual_information, single_orbital_entropy,
                               two_orbital_entropy)

ALG_TOL = 1e-10


# --- a small synthetic two-component system ------------------------------------------------
@pytest.fixture(scope="module")
def system():
    """AO factors, a 2c one-electron Hamiltonian, orthonormal spinors and spaces.

    The starting orbitals are the eigenvectors of the one-electron Hamiltonian — a core
    guess, which is what a real calculation starts from. A *random* unitary is a much harsher
    starting point than anything physical and turns a convergence test into a global
    optimization test.
    """
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
    return factors, h_ao, np.ascontiguousarray(c0), spaces, 2      # last: active electrons


def solve_cas(factors, h_ao, coeff, spaces, nelec):
    """Exact CI in the active space; returns (energy, gamma, Gamma, CASIntegrals)."""
    ints = CASIntegrals.build(factors, h_ao, coeff, spaces, e_nuc=0.0)
    occ = list(itertools.combinations(range(spaces.n_active), nelec))
    dets = Determinants.from_occupations(occ, spaces.n_active)
    mat = hamiltonian_matrix(dets, ints.h_active_effective(), ints.active_eri()).toarray()
    w, v = np.linalg.eigh(mat)
    g1, g2 = rdm12(dets, v[:, 0])
    return w[0] + ints.e_core, g1, g2, ints


# --- orbital spaces --------------------------------------------------------------------------
def test_spaces_must_partition():
    with pytest.raises(ValueError, match="partition"):
        OrbitalSpaces(inactive=[0, 1], active=[1, 2], virtual=[3], n_orb=4)


def test_redundant_rotations_are_excluded():
    """Inactive-inactive and virtual-virtual rotations change no observable and must never
    become parameters — including them makes the Hessian singular by construction."""
    sp = OrbitalSpaces.from_counts(2, 3, 8)
    rows, cols = sp.rotation_pairs()
    assert rows.size == 3 * 2 + 3 * 2 + 3 * 3       # act-inact, virt-inact, virt-act
    pairs = set(zip(rows.tolist(), cols.tolist()))
    for i, j in itertools.combinations(sp.inactive.tolist(), 2):
        assert (i, j) not in pairs and (j, i) not in pairs
    for a, b in itertools.combinations(sp.virtual.tolist(), 2):
        assert (a, b) not in pairs and (b, a) not in pairs
    # active-active is opt-in
    assert sp.rotation_pairs(active_active=True)[0].size == rows.size + 3


def test_unitary_from_antihermitian_is_exactly_unitary():
    rng = np.random.default_rng(11)
    n = 7
    a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    kappa = a - a.conj().T
    u = unitary_from_antihermitian(kappa)
    assert np.max(np.abs(u.conj().T @ u - np.eye(n))) < 1e-14
    from scipy.linalg import expm
    assert np.max(np.abs(u - expm(kappa))) < 1e-10


def test_non_antihermitian_generator_warns(kuiva_caplog):
    with kuiva_caplog.at_level("WARNING"):
        unitary_from_antihermitian(np.eye(3, dtype=complex))
    assert any("anti-Hermitian" in r.message for r in kuiva_caplog.records)


# --- energy and generalized Fock ---------------------------------------------------------------
def test_cas_energy_matches_the_ci_eigenvalue(system):
    factors, h_ao, coeff, spaces, nelec = system
    e_ci, g1, g2, ints = solve_cas(factors, h_ao, coeff, spaces, nelec)
    assert abs(cas_energy(ints, g1, g2) - e_ci) < 1e-10


def _total_rdms(gamma, gamma2, spaces, n):
    """Total D and Gamma over all orbitals, for the general generalized-Fock formula."""
    d = np.zeros((n, n), dtype=complex)
    g = np.zeros((n,) * 4, dtype=complex)
    inact, act = spaces.inactive, spaces.active
    d[inact, inact] = 1.0
    d[np.ix_(act, act)] = gamma
    for i in inact:
        for j in inact:
            g[i, i, j, j] += 1.0
            g[i, j, j, i] -= 1.0
    for i in inact:
        g[np.ix_([i], [i], act, act)] += gamma[None, None, :, :]
        g[np.ix_(act, act, [i], [i])] += gamma[:, :, None, None]
        g[np.ix_([i], act, act, [i])] -= gamma.T[None, :, :, None]
        g[np.ix_(act, [i], [i], act)] -= gamma[:, None, None, :]
    g[np.ix_(act, act, act, act)] += gamma2
    return d, g


def test_generalized_fock_specialization_matches_the_general_formula(system):
    """The efficient form uses only integrals with an active index; the general form needs
    the full ``n^4`` set. They must agree exactly — this is what licenses the fast path."""
    factors, h_ao, coeff, spaces, nelec = system
    _, g1, g2, ints = solve_cas(factors, h_ao, coeff, spaces, nelec)
    n = ints.n_orb
    b = transform_3c(factors, coeff, coeff)
    eri = assemble_4c(b, b)
    h_mo = transform_1e(h_ao, coeff)
    d, g = _total_rdms(g1, g2, spaces, n)
    f_general = np.einsum("pr,qr->pq", d, h_mo) + np.einsum("prst,qrst->pq", g, eri)
    f_special = generalized_fock(ints, g1, g2, _active_fock(ints, factors, coeff, g1))
    assert np.max(np.abs(f_special - f_general)) < 1e-9


def test_total_rdm_energy_matches(system):
    """Cross-check of the fixture itself: the energy from full MO integrals and total RDMs."""
    factors, h_ao, coeff, spaces, nelec = system
    e_ci, g1, g2, ints = solve_cas(factors, h_ao, coeff, spaces, nelec)
    b = transform_3c(factors, coeff, coeff)
    d, g = _total_rdms(g1, g2, spaces, ints.n_orb)
    e = (np.einsum("pq,pq->", d, transform_1e(h_ao, coeff)) +
         0.5 * np.einsum("pqrs,pqrs->", assemble_4c(b, b), g))
    assert abs(e.real - e_ci) < 1e-9


# --- the gradient ------------------------------------------------------------------------------
def test_gradient_matches_finite_differences(system):
    """The definitive check. RDMs are held fixed while the orbitals rotate, which is exactly
    the function the analytic gradient differentiates."""
    factors, h_ao, coeff, spaces, nelec = system
    _, g1, g2, ints = solve_cas(factors, h_ao, coeff, spaces, nelec)
    f_act = _active_fock(ints, factors, coeff, g1)
    gmat = orbital_gradient(ints, g1, g2, f_act)
    rows, cols = spaces.rotation_pairs(active_active=True)
    analytic = 2.0 * np.conj(_pack(gmat, rows, cols))

    def energy_at(vec):
        u = unitary_from_antihermitian(_unpack(vec, rows, cols, ints.n_orb))
        return cas_energy(CASIntegrals.build(factors, h_ao, coeff @ u, spaces), g1, g2)

    eps = 1e-5
    numeric = np.zeros_like(analytic)
    for m in range(analytic.size):
        for part in (0, 1):
            d = np.zeros_like(analytic)
            d[m] = eps if part == 0 else 1j * eps
            val = (energy_at(d) - energy_at(-d)) / (2 * eps)
            numeric[m] += val if part == 0 else 1j * val
    assert np.max(np.abs(analytic - numeric)) < 1e-6
    assert np.max(np.abs(analytic)) > 1e-3          # a non-trivial gradient


def test_gradient_matrix_is_antihermitian(system):
    factors, h_ao, coeff, spaces, nelec = system
    _, g1, g2, ints = solve_cas(factors, h_ao, coeff, spaces, nelec)
    g = orbital_gradient(ints, g1, g2, _active_fock(ints, factors, coeff, g1))
    assert np.max(np.abs(g + g.conj().T)) < ALG_TOL


def test_redundant_rotations_do_not_change_the_energy(system):
    """Inactive-inactive and virtual-virtual rotations are exactly redundant."""
    factors, h_ao, coeff, spaces, nelec = system
    e0, g1, g2, _ = solve_cas(factors, h_ao, coeff, spaces, nelec)
    rng = np.random.default_rng(21)
    n = spaces.n_orb
    kappa = np.zeros((n, n), dtype=complex)
    for block in (spaces.inactive, spaces.virtual):
        a = rng.standard_normal((block.size, block.size)) * 0.3
        a = a - a.T
        kappa[np.ix_(block, block)] = a
    u = unitary_from_antihermitian(kappa)
    # rotating inactive among themselves leaves the CI problem invariant, so re-solve
    e1, _, _, _ = solve_cas(factors, h_ao, coeff @ u, spaces, nelec)
    assert abs(e1 - e0) < 1e-9


def test_active_active_gradient_vanishes_for_a_converged_ci(system):
    """The generalized Brillouin condition: with the CI solved in the *full* active space,
    active-active orbital rotations are redundant and their gradient must vanish."""
    factors, h_ao, coeff, spaces, nelec = system
    _, g1, g2, ints = solve_cas(factors, h_ao, coeff, spaces, nelec)
    g = orbital_gradient(ints, g1, g2, _active_fock(ints, factors, coeff, g1))
    act = spaces.active
    assert np.max(np.abs(g[np.ix_(act, act)])) < 1e-8


def test_diagonal_hessian_is_positive_and_finite(system):
    factors, h_ao, coeff, spaces, nelec = system
    _, g1, g2, ints = solve_cas(factors, h_ao, coeff, spaces, nelec)
    rows, cols = spaces.rotation_pairs()
    hd = diagonal_hessian(ints, g1, _active_fock(ints, factors, coeff, g1), rows, cols)
    assert hd.shape == rows.shape
    assert np.all(hd > 0.0) and np.all(np.isfinite(hd))


# --- the optimizer -------------------------------------------------------------------------------
def test_optimizer_lowers_the_energy_and_converges(system):
    """End to end: alternating exact CI and orbital steps must reach a stationary point."""
    factors, h_ao, coeff, spaces, nelec = system

    def ci_solver(ints):
        occ = list(itertools.combinations(range(spaces.n_active), nelec))
        dets = Determinants.from_occupations(occ, spaces.n_active)
        mat = hamiltonian_matrix(dets, ints.h_active_effective(),
                                 ints.active_eri()).toarray()
        w, v = np.linalg.eigh(mat)
        g1, g2 = rdm12(dets, v[:, 0])
        return w[0] + ints.e_core, g1, g2

    res = optimize_orbitals(factors, h_ao, coeff, spaces, ci_solver, max_iter=300,
                            conv_grad=1e-5, report=False)
    assert res.converged, "gradient norm {:.2e}".format(res.grad_norm)
    assert res.energy < res.history[0] - 1e-6           # it actually went downhill
    assert res.grad_norm < 1e-5
    # Monotone descent is guaranteed by the accept/reject trust region, not incidental.
    assert np.all(np.diff(res.history) <= 1e-8)
    assert np.max(np.abs(res.coeff.conj().T @ res.coeff - np.eye(spaces.n_orb))) < 1e-10


def test_rejection_retry_recomputes_nothing(system):
    """After reject(), the next step() at the same iterate reuses the cached gradient,
    Hessian and Krylov subspace: zero new Hessian-vector products, zero active-Fock
    builds (the expensive half of a gradient), and a step that respects the shrunken
    trust radius."""
    import kuiva.mcscf.orbopt as orbopt_module
    factors, h_ao, coeff, spaces, nelec = system
    energy, gamma, gamma2, ints = solve_cas(factors, h_ao, coeff, spaces, nelec)
    opt = OrbitalOptimizer(spaces, mode="second-order")

    n_grad = [0]
    original_fock = orbopt_module._active_fock_ao

    def counting_fock(*args, **kwargs):
        n_grad[0] += 1
        return original_fock(*args, **kwargs)

    orbopt_module._active_fock_ao = counting_fock
    try:
        step1 = opt.step(ints, gamma, gamma2, factors, coeff, energy=energy, h_ao=h_ao)
        hvp_after_first = opt.n_hessian_matvec
        assert n_grad[0] == 1 and hvp_after_first > 0
        opt.reject()
        trust = opt.trust
        step2 = opt.step(ints, gamma, gamma2, factors, coeff, energy=energy, h_ao=h_ao)
        assert n_grad[0] == 1                           # gradient reused, not rebuilt
        assert opt.n_hessian_matvec == hvp_after_first  # subspace reused: zero products
        assert step2.max_rotation <= trust * (1.0 + 1e-12)
        assert step2.max_rotation < step1.max_rotation  # the trust shrink actually bit
        # A fresh iterate must miss the cache: accept clears it.
        opt.accept()
        c2 = np.ascontiguousarray(coeff @ step2.unitary)
        _, g1b, g2b, ints2 = solve_cas(factors, h_ao, c2, spaces, nelec)
        opt.step(ints2, g1b, g2b, factors, c2, energy=energy, h_ao=h_ao)
        assert n_grad[0] == 2
    finally:
        orbopt_module._active_fock_ao = original_fock


def test_gradient_is_built_once_per_iterate(system):
    """The driver evaluates the gradient at an accepted iterate for L-BFGS, and the next
    step() needs it at the same iterate: the memo must make that one build, not two.
    Counted on the active-Fock J/K build, the expensive half of a gradient."""
    import kuiva.mcscf.orbopt as orbopt_module
    factors, h_ao, coeff, spaces, nelec = system

    def ci_solver(ints):
        occ = list(itertools.combinations(range(spaces.n_active), nelec))
        dets = Determinants.from_occupations(occ, spaces.n_active)
        w, v = np.linalg.eigh(hamiltonian_matrix(dets, ints.h_active_effective(),
                                                 ints.active_eri()).toarray())
        g1, g2 = rdm12(dets, v[:, 0])
        return w[0] + ints.e_core, g1, g2

    n_fock = [0]
    original_fock = orbopt_module._active_fock_ao

    def counting_fock(*args, **kwargs):
        n_fock[0] += 1
        return original_fock(*args, **kwargs)

    orbopt_module._active_fock_ao = counting_fock
    try:
        res = optimize_orbitals(factors, h_ao, coeff, spaces, ci_solver, max_iter=6,
                                conv_grad=1e-9, mode="quasi-newton", report=False)
    finally:
        orbopt_module._active_fock_ao = original_fock
    # One build per distinct iterate: the starting one plus one per accepted step.
    accepted = res.n_iterations - res.n_rejected
    assert n_fock[0] == accepted + 1, \
        "{} active-Fock builds for {} accepted iterations".format(n_fock[0], accepted)


def test_optimizer_is_stationary_at_its_own_solution(system):
    """Restarting from the converged orbitals must take a vanishing step."""
    factors, h_ao, coeff, spaces, nelec = system

    def ci_solver(ints):
        occ = list(itertools.combinations(range(spaces.n_active), nelec))
        dets = Determinants.from_occupations(occ, spaces.n_active)
        w, v = np.linalg.eigh(hamiltonian_matrix(dets, ints.h_active_effective(),
                                                 ints.active_eri()).toarray())
        g1, g2 = rdm12(dets, v[:, 0])
        return w[0] + ints.e_core, g1, g2

    res = optimize_orbitals(factors, h_ao, coeff, spaces, ci_solver, max_iter=300,
                            conv_grad=1e-5, report=False)
    ints = CASIntegrals.build(factors, h_ao, res.coeff, spaces)
    opt = OrbitalOptimizer(spaces)
    step = opt.step(ints, res.gamma, res.gamma2, factors, res.coeff, h_ao=h_ao)
    assert step.grad_norm < 1e-5
    assert step.max_rotation < 1e-4


# --- the exact Hessian ---------------------------------------------------------------------------
def test_hessian_equals_the_numerical_hessian_and_is_symmetric(system):
    """The definitive check: the analytic Hessian-vector product, built densely, must equal
    the numerical second-derivative matrix of the energy **and** be exactly symmetric.

    Both halves matter. The one-index transformed Fock builds are what the first half tests.
    The second half tests the chart-curvature correction: without it the operator is the
    Jacobian of the gradient *field*, which is asymmetric by a term of the order of the
    gradient — and an asymmetric operator silently wrecks the Davidson solve built on it,
    while passing every quadratic-form check (that term has zero quadratic form).
    """
    factors, h_ao, coeff, spaces, nelec = system
    _, g1, g2, ints = solve_cas(factors, h_ao, coeff, spaces, nelec)
    rows, cols = spaces.rotation_pairs()
    m = rows.size
    hess = OrbitalHessian(ints, factors, h_ao, coeff, g1, g2, rows, cols)

    def to_c(v):
        return v[:m] + 1j * v[m:]

    def to_r(v):
        return np.concatenate([v.real, v.imag])

    a = np.zeros((2 * m, 2 * m))
    for k in range(2 * m):
        e = np.zeros(2 * m)
        e[k] = 1.0
        a[:, k] = to_r(hess(to_c(e)))

    def energy_at(v):
        u = unitary_from_antihermitian(_unpack(to_c(v), rows, cols, spaces.n_orb))
        return cas_energy(CASIntegrals.build(factors, h_ao, coeff @ u, spaces), g1, g2)

    eps = 1e-4
    b = np.zeros((2 * m, 2 * m))
    for i in range(2 * m):
        for j in range(i, 2 * m):
            ei = np.zeros(2 * m); ei[i] = eps
            ej = np.zeros(2 * m); ej[j] = eps
            b[i, j] = b[j, i] = (energy_at(ei + ej) - energy_at(ei - ej)
                                 - energy_at(-ei + ej) + energy_at(-ei - ej)) / (4 * eps ** 2)
    scale = max(np.max(np.abs(b)), 1.0)
    assert np.max(np.abs(a - a.T)) < 1e-10 * scale        # exactly symmetric
    assert np.max(np.abs(a - b)) < 1e-5 * scale           # and it is the right matrix
    assert np.max(np.abs(b)) > 1.0                        # a non-trivial Hessian


def test_hessian_matches_the_gradient_derivative_up_to_chart_curvature(system):
    """The raw one-index transformation is the derivative of the gradient field; the operator
    differs from it by exactly ``-1/2 [kappa, conj(G)]``.

    Pinning the curvature term separately is what makes the two halves of the construction
    independently debuggable — a mistake in the Fock builds and a mistake in the correction
    look identical when only the total is checked.
    """
    factors, h_ao, coeff, spaces, nelec = system
    _, g1, g2, ints = solve_cas(factors, h_ao, coeff, spaces, nelec)
    rows, cols = spaces.rotation_pairs()
    hess = OrbitalHessian(ints, factors, h_ao, coeff, g1, g2, rows, cols)
    gmat = orbital_gradient(ints, g1, g2, _active_fock(ints, factors, coeff, g1))

    def grad_at(vec):
        u = unitary_from_antihermitian(_unpack(vec, rows, cols, spaces.n_orb))
        cc = np.ascontiguousarray(coeff @ u)
        it = CASIntegrals.build(factors, h_ao, cc, spaces)
        gm = orbital_gradient(it, g1, g2, _active_fock(it, factors, cc, g1))
        return 2.0 * np.conj(_pack(gm, rows, cols))

    rng = np.random.default_rng(51)
    d = rng.standard_normal(rows.size) + 1j * rng.standard_normal(rows.size)
    d /= np.linalg.norm(d)
    eps = 1e-5
    field_derivative = (grad_at(eps * d) - grad_at(-eps * d)) / (2 * eps)
    kappa = _unpack(d, rows, cols, spaces.n_orb)
    g_dual = np.conj(gmat)
    curv = -0.5 * (kappa @ g_dual - g_dual @ kappa)
    expected = field_derivative - 2.0 * _pack(curv, rows, cols)
    assert np.max(np.abs(hess(d) - expected)) < 1e-6 * max(np.max(np.abs(expected)), 1.0)


def test_hessian_quadratic_form_matches_the_second_derivative(system):
    factors, h_ao, coeff, spaces, nelec = system
    _, g1, g2, ints = solve_cas(factors, h_ao, coeff, spaces, nelec)
    rows, cols = spaces.rotation_pairs()
    hess = OrbitalHessian(ints, factors, h_ao, coeff, g1, g2, rows, cols)
    rng = np.random.default_rng(52)
    d = rng.standard_normal(rows.size) + 1j * rng.standard_normal(rows.size)
    d /= np.linalg.norm(d)

    def energy_at(v):
        u = unitary_from_antihermitian(_unpack(v, rows, cols, spaces.n_orb))
        return cas_energy(CASIntegrals.build(factors, h_ao, coeff @ u, spaces), g1, g2)

    eps = 1e-4
    numeric = (energy_at(eps * d) - 2 * energy_at(0 * d) + energy_at(-eps * d)) / eps ** 2
    assert abs(np.real(np.vdot(hess(d), d)) - numeric) < 1e-4 * max(abs(numeric), 1.0)


def test_hessian_square_route_matches_the_ao_route(system):
    """The fixed-factor (MO-square) product against the blocked AO product.

    The two routes are one operator through two formulations; nothing but roundoff may
    separate them, on the products and on the exact diagonal alike. This is what keeps
    the AO fallback honest now that the small-system default is the square route.
    """
    factors, h_ao, coeff, spaces, nelec = system
    energy, gamma, gamma2, ints = solve_cas(factors, h_ao, coeff, spaces, nelec)
    rows, cols = spaces.rotation_pairs()
    hess_sq = OrbitalHessian(ints, factors, h_ao, coeff, gamma, gamma2, rows, cols,
                             mo_square=True)
    hess_ao = OrbitalHessian(ints, factors, h_ao, coeff, gamma, gamma2, rows, cols,
                             mo_square=False)
    assert hess_sq.mo_square and not hess_ao.mo_square
    rng = np.random.default_rng(59)
    for _ in range(3):
        v = (rng.standard_normal(rows.size) + 1j * rng.standard_normal(rows.size))
        out_sq = hess_sq.matvec(v)
        out_ao = hess_ao.matvec(v)
        scale = float(np.max(np.abs(out_ao))) + 1.0
        assert np.max(np.abs(out_sq - out_ao)) < 1e-10 * scale
    d_sq = hess_sq.exact_diagonal()
    d_ao = hess_ao.exact_diagonal()
    assert np.max(np.abs(d_sq - d_ao)) < 1e-10 * (float(np.max(np.abs(d_ao))) + 1.0)


def test_square_route_sparse_and_dense_kappa_forms_agree(system):
    """The kappa-sparse product forms against the dense general forms, and the
    active-active guard: with active-active rotations as parameters the sparse active
    form's zero-block assumption fails, so the guard must route around it — checked by
    comparing against the AO route, which knows nothing of either."""
    factors, h_ao, coeff, spaces, nelec = system
    energy, gamma, gamma2, ints = solve_cas(factors, h_ao, coeff, spaces, nelec)
    rng = np.random.default_rng(60)
    rows, cols = spaces.rotation_pairs()
    hess = OrbitalHessian(ints, factors, h_ao, coeff, gamma, gamma2, rows, cols,
                          mo_square=True)
    assert hess._occ_prefix and not hess._has_act_act    # the fast forms are live
    v = rng.standard_normal(rows.size) + 1j * rng.standard_normal(rows.size)
    fast = hess.matvec(v)
    hess._occ_prefix = False                             # force the dense general forms
    dense = hess.matvec(v)
    scale = float(np.max(np.abs(dense))) + 1.0
    assert np.max(np.abs(fast - dense)) < 1e-11 * scale
    # Active-active rotations: the guard must detect them and take the dense active form.
    rows2, cols2 = spaces.rotation_pairs(active_active=True)
    h_sq = OrbitalHessian(ints, factors, h_ao, coeff, gamma, gamma2, rows2, cols2,
                          mo_square=True)
    h_ao_route = OrbitalHessian(ints, factors, h_ao, coeff, gamma, gamma2, rows2, cols2,
                                mo_square=False)
    assert h_sq._has_act_act
    v2 = rng.standard_normal(rows2.size) + 1j * rng.standard_normal(rows2.size)
    out_sq = h_sq.matvec(v2)
    out_ao = h_ao_route.matvec(v2)
    scale = float(np.max(np.abs(out_ao))) + 1.0
    assert np.max(np.abs(out_sq - out_ao)) < 1e-10 * scale


def test_hessian_square_sizing_is_exact(system):
    """Two-sided pin of the fixed-factor route's sizing against the arrays it holds."""
    from kuiva.mcscf.orbopt import hessian_square_memory_gb
    factors, h_ao, coeff, spaces, nelec = system
    energy, gamma, gamma2, ints = solve_cas(factors, h_ao, coeff, spaces, nelec)
    rows, cols = spaces.rotation_pairs()
    hess = OrbitalHessian(ints, factors, h_ao, coeff, gamma, gamma2, rows, cols,
                          mo_square=True)
    held = (hess.b_full.nbytes + hess.b_occ_t.nbytes
            + hess.b_i_occ_flat.nbytes + hess.b_a_occ_flat.nbytes
            + hess.b_p_i_flat.nbytes + hess.b_p_a_flat.nbytes)
    n_occ = spaces.n_inactive + spaces.n_active
    got = hessian_square_memory_gb(factors.naux, spaces.n_orb, n_occ) * 1024.0 ** 3
    assert got == pytest.approx(held, rel=0.0, abs=0.5)


def test_exact_diagonal_matches_the_matvec_diagonal(system):
    """The analytic Hessian diagonal against the one extracted from matvec.

    The matvec is verified independently against the numerical Hessian, so this pins the
    per-class formulas in :meth:`OrbitalHessian.exact_diagonal` — an error there cannot
    pass. The complex-linear diagonal is extracted as
    ``(matvec(e_j)_j + (-i) matvec(i e_j)_j) / 2``, which cancels the anti-linear part
    exactly; the preconditioner is that mean curvature by construction.
    """
    factors, h_ao, coeff, spaces, nelec = system
    energy, gamma, gamma2, ints = solve_cas(factors, h_ao, coeff, spaces, nelec)
    rows, cols = spaces.rotation_pairs()
    hess = OrbitalHessian(ints, factors, h_ao, coeff, gamma, gamma2, rows, cols)
    hd = hess.exact_diagonal()
    assert hd.shape == rows.shape and hd.dtype == np.float64
    scale = float(np.max(np.abs(hd))) + 1.0
    for j in range(rows.size):
        e_j = np.zeros(rows.size, dtype=np.complex128)
        e_j[j] = 1.0
        mv1 = hess.matvec(e_j)[j]
        mv2 = hess.matvec(1j * e_j)[j]
        diag_num = 0.5 * (mv1 + (-1j) * mv2)
        assert abs(diag_num.imag) < 1e-9 * scale
        assert abs(hd[j] - diag_num.real) < 1e-9 * scale, \
            "pair ({}, {}) [j = {}]: exact {} vs matvec {}".format(
                rows[j], cols[j], j, hd[j], diag_num.real)
    # Cached: a second call is the same object, not a recomputation.
    assert hess.exact_diagonal() is hd


def test_ritz_recycling_mechanics():
    """The recycling machinery: Ritz vectors come back normalized, a seeded solve reaches
    the same Newton step, and a redundant seed is dropped for free.

    Whether recycling *pays* is a property of the trajectory, not of the mechanics, and is
    measured on the real benchmark — a synthetic problem here would be exactly the
    over-tuning the module's own validation record warns against.
    """
    rng = np.random.default_rng(58)
    m = 24
    q, _ = np.linalg.qr(rng.standard_normal((2 * m, 2 * m)))
    hmat = q @ np.diag(np.geomspace(0.05, 50.0, 2 * m)) @ q.T

    def to_c(v):
        return v[:m] + 1j * v[m:]

    def to_r(v):
        return np.concatenate([v.real, v.imag])

    hdiag = np.abs(np.diag(hmat))[:m]
    grad = to_c(rng.standard_normal(2 * m))

    calls = [0]

    def hvp(x):
        calls[0] += 1
        return to_c(hmat @ to_r(x))

    first = augmented_hessian_step(grad, hvp, hdiag, tol=1e-8, max_iter=200,
                                   max_subspace=200, n_ritz_out=4)
    assert first.converged
    assert first.ritz_vectors and len(first.ritz_vectors) <= 4
    for vec in first.ritz_vectors:
        assert abs(float(np.real(np.vdot(vec, vec))) - 1.0) < 1e-10   # normalized
    # A seeded solve of a nearby problem converges to the same shifted-model solution.
    grad2 = grad + 0.05 * to_c(rng.standard_normal(2 * m))
    cold = augmented_hessian_step(grad2, hvp, hdiag, tol=1e-8, max_iter=200,
                                  max_subspace=200)
    seeded = augmented_hessian_step(grad2, hvp, hdiag, tol=1e-8, max_iter=200,
                                    max_subspace=200, recycle=first.ritz_vectors)
    assert seeded.converged and cold.converged
    assert np.linalg.norm(seeded.step - cold.step) < 1e-5 * np.linalg.norm(cold.step)
    # A seed the basis already contains costs nothing: recycling the warm-start vector
    # itself must not change the product count of a guess-only solve.
    calls[0] = 0
    guess_only = augmented_hessian_step(grad2, hvp, hdiag, tol=1e-8, max_iter=200,
                                        max_subspace=200, guess=first.step)
    n_guess = calls[0]
    calls[0] = 0
    with_dup = augmented_hessian_step(grad2, hvp, hdiag, tol=1e-8, max_iter=200,
                                      max_subspace=200, guess=first.step,
                                      recycle=[first.step /
                                               np.linalg.norm(first.step)])
    assert calls[0] == n_guess
    assert np.linalg.norm(with_dup.step - guess_only.step) < 1e-8


def test_augmented_hessian_solves_a_known_quadratic():
    """Against a dense reference: the AH step must match the Newton step of an explicit
    positive-definite quadratic model."""
    rng = np.random.default_rng(53)
    m = 12
    a = rng.standard_normal((2 * m, 2 * m))
    hmat = a @ a.T + 5.0 * np.eye(2 * m)               # symmetric positive definite

    def to_c(v):
        return v[:m] + 1j * v[m:]

    def to_r(v):
        return np.concatenate([v.real, v.imag])

    grad = to_c(rng.standard_normal(2 * m))
    hvp = lambda x: to_c(hmat @ to_r(x))               # noqa: E731
    hdiag = np.diag(hmat)[:m] + np.diag(hmat)[m:]
    res = augmented_hessian_step(grad, hvp, hdiag, tol=1e-8, max_iter=80, max_subspace=60)
    # The defining property, exact whatever the shift turns out to be: the AH step solves the
    # *level-shifted* Newton equations (H - mu) x = -g. Comparing against the unshifted Newton
    # step would only test how small mu happened to be.
    assert res.converged
    lhs = hmat @ to_r(res.step) - res.eigenvalue * to_r(res.step)
    assert np.linalg.norm(lhs + to_r(grad)) < 1e-6 * np.linalg.norm(to_r(grad))
    assert res.eigenvalue < 0.0                        # a genuine (negative) level shift
    assert float(np.real(np.vdot(res.step, grad))) < 0.0        # downhill


def test_augmented_hessian_handles_indefinite_curvature():
    """With a negative eigenvalue in H the plain Newton step points uphill; the AH step must
    still be a descent direction — that is the entire reason for the augmented formulation."""
    rng = np.random.default_rng(54)
    m = 8
    hmat = np.diag(np.concatenate([[-2.0, -0.5], rng.uniform(1.0, 5.0, 2 * m - 2)]))

    def to_c(v):
        return v[:m] + 1j * v[m:]

    def to_r(v):
        return np.concatenate([v.real, v.imag])

    grad = to_c(rng.standard_normal(2 * m))
    res = augmented_hessian_step(grad, lambda x: to_c(hmat @ to_r(x)),
                                 np.abs(np.diag(hmat))[:m], tol=1e-8, max_iter=60,
                                 max_subspace=40)
    assert res.eigenvalue < -2.0                       # shifted below the lowest curvature
    assert float(np.real(np.vdot(res.step, grad))) < 0.0


def test_second_order_mode_converges_faster(system):
    """The point of the whole exercise."""
    factors, h_ao, coeff, spaces, nelec = system

    def ci_solver(ints):
        occ = list(itertools.combinations(range(spaces.n_active), nelec))
        dets = Determinants.from_occupations(occ, spaces.n_active)
        w, v = np.linalg.eigh(hamiltonian_matrix(dets, ints.h_active_effective(),
                                                 ints.active_eri()).toarray())
        g1, g2 = rdm12(dets, v[:, 0])
        return w[0] + ints.e_core, g1, g2

    runs = {}
    for mode in ("quasi-newton", "second-order", "auto"):
        runs[mode] = optimize_orbitals(factors, h_ao, coeff, spaces, ci_solver, max_iter=300,
                                       conv_grad=1e-6, mode=mode, report=False)
    assert all(r.converged for r in runs.values())
    # all modes must find the same stationary point
    for mode, r in runs.items():
        assert abs(r.energy - runs["second-order"].energy) < 1e-7, mode
    assert runs["second-order"].n_iterations < 0.5 * runs["quasi-newton"].n_iterations
    assert runs["auto"].n_iterations <= runs["quasi-newton"].n_iterations


def test_quasi_newton_mode_needs_no_h_ao(system):
    """The cheap path must remain usable without the machinery the Hessian needs."""
    factors, h_ao, coeff, spaces, nelec = system
    _, g1, g2, ints = solve_cas(factors, h_ao, coeff, spaces, nelec)
    opt = OrbitalOptimizer(spaces, mode="quasi-newton")
    step = opt.step(ints, g1, g2, factors, coeff)
    assert not step.second_order
    with pytest.raises(ValueError, match="needs h_ao"):
        OrbitalOptimizer(spaces, mode="second-order").step(ints, g1, g2, factors, coeff)


def test_unknown_mode_rejected():
    with pytest.raises(ValueError, match="unknown optimizer mode"):
        OrbitalOptimizer(OrbitalSpaces.from_counts(1, 2, 6), mode="newton-raphson")


def test_augmented_hessian_respects_the_trust_radius_by_shifting():
    """An over-long Newton step must be shortened by raising the level shift, not by scaling
    the converged step — a truncated step solves no model at all."""
    rng = np.random.default_rng(55)
    m = 10
    a = rng.standard_normal((2 * m, 2 * m))
    hmat = a @ a.T + 0.05 * np.eye(2 * m)              # nearly singular: a long Newton step

    def to_c(v):
        return v[:m] + 1j * v[m:]

    def to_r(v):
        return np.concatenate([v.real, v.imag])

    grad = to_c(rng.standard_normal(2 * m))
    hvp = lambda x: to_c(hmat @ to_r(x))               # noqa: E731
    hdiag = np.abs(np.diag(hmat))[:m]
    free = augmented_hessian_step(grad, hvp, hdiag, tol=1e-8, max_iter=80)
    trust = 0.1 * float(np.max(np.abs(free.step)))
    # step_tol=0 asks for the fully resolved shifted step; the default trust-aware
    # truncation deliberately trades that exactness for Hessian-vector products and is
    # tested separately below.
    capped = augmented_hessian_step(grad, hvp, hdiag, tol=1e-8, max_iter=80, trust=trust,
                                    step_tol=0.0)
    assert float(np.max(np.abs(capped.step))) <= trust * 1.05
    assert capped.shift > 0.0 and free.shift == 0.0     # shortened by shifting, not scaling
    # ... and it is still the *exact* solution of the shifted model, i.e. second-order
    # informed rather than a truncated direction.
    x = to_r(capped.step)
    resid = hmat @ x + capped.shift * x - capped.eigenvalue * x + to_r(grad)
    assert np.linalg.norm(resid) < 1e-4 * np.linalg.norm(to_r(grad))
    assert float(np.real(np.vdot(capped.step, grad))) < 0.0


def test_augmented_hessian_trust_truncation_saves_products():
    """The default step_tol stops a trust-capped solve once the shifted step stabilizes.

    The shifted step is what the caller takes, so refining the subspace past the point
    where it has stopped moving only polishes the part of the direction the shift then
    discards. The truncated solve must cost fewer Hessian-vector products, stay inside the
    trust radius, stay downhill, and land near the fully resolved shifted step.
    """
    rng = np.random.default_rng(56)
    m = 40
    a = rng.standard_normal((2 * m, 2 * m))
    hmat = a @ a.T + 0.05 * np.eye(2 * m)

    def to_c(v):
        return v[:m] + 1j * v[m:]

    def to_r(v):
        return np.concatenate([v.real, v.imag])

    grad = 10.0 * to_c(rng.standard_normal(2 * m))     # far from convergence: capped regime
    hvp = lambda x: to_c(hmat @ to_r(x))               # noqa: E731
    hdiag = np.abs(np.diag(hmat))[:m]
    exact = augmented_hessian_step(grad, hvp, hdiag, tol=1e-10, max_iter=150,
                                   max_subspace=150, trust=1e-3, step_tol=0.0)
    truncated = augmented_hessian_step(grad, hvp, hdiag, tol=1e-10, max_iter=150,
                                       max_subspace=150, trust=1e-3)
    assert truncated.n_matvec < exact.n_matvec
    assert float(np.max(np.abs(truncated.step))) <= 1e-3 * 1.05
    assert float(np.real(np.vdot(truncated.step, grad))) < 0.0
    # Within the advertised tolerance of the resolved shifted step.
    diff = np.linalg.norm(truncated.step - exact.step)
    assert diff <= 0.35 * np.linalg.norm(exact.step)


def test_augmented_hessian_subspace_reuse_costs_no_products():
    """A rejection retry re-solves in the returned subspace with zero Hessian-vector
    products, and its trust-shifted step matches a fresh solve at the same trust radius to
    the accuracy the subspace supports."""
    rng = np.random.default_rng(57)
    m = 16
    a = rng.standard_normal((2 * m, 2 * m))
    hmat = a @ a.T + 0.5 * np.eye(2 * m)

    def to_c(v):
        return v[:m] + 1j * v[m:]

    def to_r(v):
        return np.concatenate([v.real, v.imag])

    grad = to_c(rng.standard_normal(2 * m))
    calls = [0]

    def hvp(x):
        calls[0] += 1
        return to_c(hmat @ to_r(x))

    first = augmented_hessian_step(grad, hvp, np.abs(np.diag(hmat))[:m], tol=1e-8,
                                   max_iter=80, trust=0.05)
    n_first = calls[0]
    retry = augmented_hessian_step(grad, hvp, np.abs(np.diag(hmat))[:m], tol=1e-8,
                                   max_iter=80, trust=0.25 * 0.05,
                                   subspace=first.subspace)
    assert calls[0] == n_first                          # zero new products
    assert retry.n_matvec == 0
    assert float(np.max(np.abs(retry.step))) <= 0.25 * 0.05 * 1.05
    assert float(np.real(np.vdot(retry.step, grad))) < 0.0
    assert retry.shift >= first.shift                   # a smaller trust needs a larger shift


def test_auto_does_not_escalate_while_the_gradient_keeps_falling():
    """The escalation policy: escalate on failure, not on proximity or on a quiet energy.

    A gradient that keeps falling means the cheap step is working, however slowly, and
    escalating there costs Hessian-vector products for nothing — measured at 118 -> 332 work
    units before this was fixed.
    """
    opt = OrbitalOptimizer(OrbitalSpaces.from_counts(2, 4, 12), mode="auto", conv_grad=1e-6)
    g = 1.0
    for _ in range(60):                                 # steady 20% reduction per iteration
        assert not opt._decide_second_order(g), "escalated while converging (|g| = %.2e)" % g
        g *= 0.8
    assert not opt._second_order


def test_auto_escalates_when_the_gradient_plateaus():
    opt = OrbitalOptimizer(OrbitalSpaces.from_counts(2, 4, 12), mode="auto", conv_grad=1e-6)
    escalated_at = None
    for i in range(60):
        if opt._decide_second_order(1e-2):              # completely stuck
            escalated_at = i
            break
    assert escalated_at is not None and escalated_at <= 2 * opt.stall_window


def test_auto_does_not_escalate_once_converged():
    """A plateau below the convergence target is success, not a stall."""
    opt = OrbitalOptimizer(OrbitalSpaces.from_counts(2, 4, 12), mode="auto", conv_grad=1e-6)
    for _ in range(60):
        assert not opt._decide_second_order(1e-9)


# --- the cheap CI ----------------------------------------------------------------------------------
def random_active_integrals(n, seed=0, spread=2.0):
    r = np.random.default_rng(seed)
    ell = r.standard_normal((3 * n, n, n))
    ell = 0.5 * (ell + ell.transpose(0, 2, 1))
    q, _ = np.linalg.qr(r.standard_normal((n, n)) + 1j * r.standard_normal((n, n)))
    b = np.einsum("mp,Pmn,nq->Ppq", q.conj(), ell, q)
    eri = 0.3 * np.tensordot(b, b, axes=([0], [0]))
    h = r.standard_normal((n, n)) + 1j * r.standard_normal((n, n))
    h = 0.5 * (h + h.conj().T) + np.diag(spread * np.arange(n))
    return h, eri


def exact_fci(h, eri, nelec, n_states=1):
    """Full CI reference. Sparse for anything but a tiny space — a dense diagonalization of
    the 12870-determinant space below costs minutes and would put this file in the slow
    suite for no benefit (default-suite timing guard)."""
    n = h.shape[0]
    dets = Determinants.from_occupations(itertools.combinations(range(n), nelec), n)
    mat = hamiltonian_matrix(dets, h, eri)
    if dets.ndet <= 400:
        w, v = np.linalg.eigh(mat.toarray())
    else:
        from scipy.sparse.linalg import eigsh
        w, v = eigsh(mat, k=max(n_states, 2), which="SA")
        order = np.argsort(w)
        w, v = w[order], v[:, order]
    return w[:n_states], v[:, :n_states], dets


def test_cheap_ci_reproduces_fci_when_the_space_is_complete():
    """With a budget beyond the full space the selection must recover FCI exactly — energy,
    1-RDM and natural occupations."""
    n, nelec = 8, 4
    h, eri = random_active_integrals(n, seed=31)
    w, v, dets = exact_fci(h, eri, nelec)
    g_exact, _ = rdm12(dets, v[:, 0], with_2rdm=False)
    res = cheap_ci(h, eri, nelec, max_determinants=10000)
    assert res.n_determinants == dets.ndet
    assert abs(res.energies[0] - w[0]) < 1e-9
    assert np.max(np.abs(res.gamma - g_exact)) < 1e-9


@pytest.mark.slow
def test_occupations_converge_faster_than_the_energy():
    """The premise of the pre-optimizer: it is judged on natural occupations, not energy.

    Slow (~27 s) because it needs a 12870-determinant FCI reference to compare against; the
    default-suite timing guard puts it in the opt-in suite rather than the default one.

    Measured on a deliberately hard random Hamiltonian, which has no locality for the
    selection to exploit and is therefore a worst case: with 3000 of 12870 determinants the
    natural occupations are right to better than 0.1 — enough to classify every spinor as
    correlated or not — while the total energy is still wrong by *tens of Eh*. If this ever
    inverts, the method has stopped being fit for its purpose and the module docstring's
    claim is false.
    """
    n, nelec = 16, 8
    h, eri = random_active_integrals(n, seed=32)
    w, v, dets = exact_fci(h, eri, nelec)
    g_exact, _ = rdm12(dets, v[:, 0], with_2rdm=False)
    occ_exact = np.sort(np.linalg.eigvalsh(g_exact))[::-1]

    errs = {}
    for budget in (200, 3000):
        res = cheap_ci(h, eri, nelec, max_determinants=budget, with_2rdm=False)
        occ, _ = res.natural_spinors()
        errs[budget] = (np.max(np.abs(occ - occ_exact)), res.energies[0] - w[0])
    assert errs[3000][0] < errs[200][0]                    # occupations improve
    assert errs[3000][0] < 0.1                             # and are qualitatively right
    assert errs[3000][1] > 1.0                             # while the energy is nowhere near


def test_reference_space_is_multireference():
    """The reference is a small CAS around the Fermi level, not a single determinant —
    the point on which a coupled-metal-centre pre-optimization succeeds or fails."""
    n, nelec = 12, 6
    h, _ = random_active_integrals(n, seed=33)
    ref = reference_determinants(np.diag(h), n, nelec, max_reference=200)
    assert ref.ndet > 1
    assert ref.n_elec == nelec
    assert ref.ndet <= 200


def test_state_averaging_is_applied():
    n, nelec = 8, 4
    h, eri = random_active_integrals(n, seed=34)
    res = cheap_ci(h, eri, nelec, n_states=3, max_determinants=10000)
    assert res.energies.size == 3
    assert np.all(np.diff(res.energies) >= -1e-9)
    assert abs(np.trace(res.gamma).real - nelec) < 1e-8
    lopsided = cheap_ci(h, eri, nelec, n_states=3, max_determinants=10000,
                        state_weights=[1.0, 0.0, 0.0])
    single = cheap_ci(h, eri, nelec, n_states=1, max_determinants=10000)
    assert np.max(np.abs(lopsided.gamma - single.gamma)) < 1e-8


def test_natural_spinors_diagonalize_the_1rdm():
    """The rotation must diagonalize gamma under the **orbital-coefficient** transformation
    law, ``gamma -> U^T gamma U*``, not under ``U^dag gamma U``.

    Coefficients and density matrices transform oppositely, so the naive ``U^dag gamma U``
    check passes for the *wrong* rotation (``V`` instead of ``conj(V)``) — it verifies that
    the eigendecomposition is an eigendecomposition, which was never in doubt, while the
    orbitals the caller actually receives are left un-diagonalized. That bug was live until
    this test was written the right way round.
    """
    n, nelec = 8, 4
    h, eri = random_active_integrals(n, seed=35)
    res = cheap_ci(h, eri, nelec, max_determinants=2000)
    occ, rot = res.natural_spinors()
    assert np.all(np.diff(occ) <= 1e-12)                       # descending
    assert np.all(occ >= -1e-12) and np.all(occ <= 1 + 1e-12)  # fermionic bounds
    assert abs(occ.sum() - nelec) < 1e-8
    assert np.max(np.abs(rot.conj().T @ rot - np.eye(n))) < 1e-12      # unitary
    g_rot = rot.T @ res.gamma @ rot.conj()
    assert np.max(np.abs(g_rot - np.diag(np.diag(g_rot)))) < 1e-10
    assert np.max(np.abs(np.real(np.diag(g_rot)) - occ)) < 1e-10


def test_solve_fixed_space_matches_a_selection_free_solve():
    """Solving in a given space must be exactly the CI in that space — no hidden selection."""
    from kuiva.mcscf.preopt import solve_fixed_space
    n, nelec = 8, 4
    h, eri = random_active_integrals(n, seed=61)
    ref = cheap_ci(h, eri, nelec, max_determinants=10000)      # complete space
    fixed = solve_fixed_space(ref.dets, h, eri)
    assert fixed.n_determinants == ref.n_determinants
    assert abs(fixed.energies[0] - ref.energies[0]) < 1e-9
    assert np.max(np.abs(fixed.gamma - ref.gamma)) < 1e-9


def test_preopt_space_policies():
    """Who owns the determinant space: the three values of ``SPACE_POLICIES``.

    A space re-selected at every point makes ``E(kappa)`` discontinuous and measurably harder
    to optimize for every step type. The default is event gating, which holds
    the space fixed *between events* — so the surface is smooth everywhere the optimizer
    evaluates it — and adopts a fresh selection only when it genuinely wins.

    The test asserts the *mechanism*: how many selections a run made, and that the two old
    spellings of ``freeze_determinants`` still mean exactly what they used to. The convergence
    benefit is a benchmark result (a recorded benchmark), not something a
    unit test should chase.

    ⚠ The key claim pinned here is that **freezing is event gating with the cadence set to
    never**: with an event interval past the iteration budget the default policy makes exactly
    the one selection the frozen policy makes. If that stops holding, the old behaviour has
    stopped being expressible.
    """
    from kuiva.mcscf.orbopt import CASIntegrals
    from kuiva.mcscf.preopt import cheap_ci as _cheap_ci

    rng = np.random.default_rng(62)
    nao = 6
    n = 2 * nao
    npair = nao * (nao + 1) // 2
    factors = ThreeIndexAO(l_packed=0.5 * rng.standard_normal((3 * nao, npair)), nao=nao,
                           origin="cholesky")
    h_ao = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    h_ao = 0.5 * (h_ao + h_ao.conj().T)
    _, c0 = np.linalg.eigh(h_ao)
    spaces = OrbitalSpaces.from_counts(2, 6, n)
    from kuiva.mcscf.preopt import preoptimize

    seen = []
    original = _cheap_ci

    import kuiva.mcscf.preopt as preopt_mod

    def spy(*a, **kw):
        res = original(*a, **kw)
        seen.append(res.dets.masks.copy())
        return res

    def selections(**kw):
        """Determinant spaces selected *during the optimization*, in order.

        The final call is the analysis solve at the returned orbitals, which always
        re-selects by design — it is not part of the surface the optimizer saw, so it
        is dropped here.
        """
        seen.clear()
        preopt_mod.cheap_ci = spy
        try:
            preoptimize(factors, h_ao, np.ascontiguousarray(c0), spaces, 4, n_states=1,
                        max_iter=6, max_determinants=60, report=False,
                        natural_spinors=False, **kw)
        finally:
            preopt_mod.cheap_ci = original
        return [tuple(m.tolist()) for m in seen[:-1]]

    frozen = selections(space_policy="frozen")
    assert len(frozen) == 1, "the frozen policy selected %d times" % len(frozen)
    assert selections(freeze_determinants=True) == frozen        # the old spelling, unchanged

    adaptive = selections(space_policy="adaptive")
    assert len(adaptive) > 1, "the adaptive policy must re-select at every point"
    assert selections(freeze_determinants=False) == adaptive     # the old spelling, unchanged

    # Freezing *is* event gating with the cadence set past the run: same single selection.
    assert selections(event_interval=10 ** 6) == frozen

    # And with a live cadence the default proposes, but never more often than the adaptive
    # policy re-selects — the whole point being that a proposal is a question, not a change.
    default = selections()
    assert 1 <= len(default) <= len(adaptive)


def test_preopt_returns_a_consistent_natural_basis():
    """End to end: the occupations of the *returned orbitals* must match the spectrum when
    the cheap CI is exact, which is the statement that the returned basis really is natural.
    """
    rng = np.random.default_rng(41)
    nao = 8
    n = 2 * nao
    npair = nao * (nao + 1) // 2
    factors = ThreeIndexAO(l_packed=0.5 * rng.standard_normal((3 * nao, npair)), nao=nao,
                           origin="cholesky")
    h_ao = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    h_ao = 0.5 * (h_ao + h_ao.conj().T)
    _, c0 = np.linalg.eigh(h_ao)
    spaces = OrbitalSpaces.from_counts(2, 8, n)
    from kuiva.mcscf.preopt import preoptimize
    res = preoptimize(factors, h_ao, np.ascontiguousarray(c0), spaces, 4, n_states=1,
                      max_iter=6, max_determinants=5000, report=False)
    assert np.max(np.abs(np.sort(res.orbital_occupation)[::-1]
                         - res.natural_occupation)) < 1e-8
    assert np.max(np.abs(res.coeff.conj().T @ res.coeff - np.eye(n))) < 1e-10


def test_bad_state_weights_rejected():
    h, eri = random_active_integrals(6, seed=36)
    with pytest.raises(ValueError, match="state_weights"):
        cheap_ci(h, eri, 3, n_states=2, state_weights=[1.0, 1.0, 1.0])


# --- entanglement -------------------------------------------------------------------------------
def test_single_orbital_entropy_analytic_values():
    gamma = np.diag([1.0, 0.0, 0.5, 0.25]).astype(complex)
    s = single_orbital_entropy(gamma)
    assert abs(s[0]) < 1e-12 and abs(s[1]) < 1e-12          # empty/full carry no entropy
    assert abs(s[2] - np.log(2.0)) < 1e-12                  # half filled is maximal
    expect = -(0.25 * np.log(0.25) + 0.75 * np.log(0.75))
    assert abs(s[3] - expect) < 1e-12


def test_mutual_information_vanishes_for_a_single_determinant():
    """A determinant is a product state over spinor modes: no orbital entanglement at all."""
    n, nelec = 8, 4
    dets = Determinants.from_occupations([tuple(range(nelec))], n)
    vec = np.ones(1, dtype=complex)
    from kuiva.ci.strings import occupation_correlations
    g, _ = rdm12(dets, vec, with_2rdm=False)
    nn = occupation_correlations(dets, vec)
    assert np.max(single_orbital_entropy(g)) < 1e-12
    assert np.max(np.abs(mutual_information(g, nn))) < 1e-12


def test_mutual_information_is_non_negative_and_symmetric():
    n, nelec = 8, 4
    h, eri = random_active_integrals(n, seed=37)
    res = cheap_ci(h, eri, nelec, max_determinants=3000)
    s1, info = res.entanglement()
    assert np.all(info >= -1e-12)
    assert np.max(np.abs(info - info.T)) < 1e-12
    assert np.all(np.diag(info) == 0.0)
    assert np.max(info) > 1e-6                       # a correlated state is entangled


def test_two_orbital_entropy_bounds():
    n, nelec = 8, 4
    h, eri = random_active_integrals(n, seed=38)
    res = cheap_ci(h, eri, nelec, max_determinants=3000)
    from kuiva.ci.strings import occupation_correlations
    s2 = two_orbital_entropy(res.gamma, res.occupation_correlation)
    s1 = single_orbital_entropy(res.gamma)
    assert np.max(np.abs(np.diag(s2) - s1)) < 1e-12          # s2_pp = s1_p
    assert np.all(s2 <= 2 * np.log(2.0) + 1e-9)              # 4-level system
    # subadditivity, both directions
    assert np.all(s2 <= s1[:, None] + s1[None, :] + 1e-9)
    assert np.all(s2 >= np.abs(s1[:, None] - s1[None, :]) - 1e-9)


def test_fiedler_ordering_recovers_a_chain():
    """A known entanglement chain must come back as a contiguous ordering (up to reversal),
    which is the property the initial MPS depends on."""
    n = 8
    perm = np.random.default_rng(39).permutation(n)
    info = np.zeros((n, n))
    for a in range(n - 1):
        info[perm[a], perm[a + 1]] = info[perm[a + 1], perm[a]] = 1.0
    order = fiedler_order(info)
    pos = np.empty(n, dtype=int)
    pos[order] = np.arange(n)
    steps = np.abs(np.diff(pos[perm]))
    assert np.all(steps == 1)


def test_disconnected_entanglement_graph_is_reported(kuiva_caplog):
    info = np.zeros((6, 6))
    info[0, 1] = info[1, 0] = 1.0
    info[3, 4] = info[4, 3] = 1.0
    with kuiva_caplog.at_level("INFO"):
        fiedler_order(info)
    assert any("disconnected" in r.message for r in kuiva_caplog.records)


# --- solver-specific columns in the shared iteration table ---------------------------------

def _exact_ci(spaces, nelec):
    """The exact active-space CI as a plain ``ci_solver(ints)`` callable."""
    def ci_solver(ints):
        occ = list(itertools.combinations(range(spaces.n_active), nelec))
        dets = Determinants.from_occupations(occ, spaces.n_active)
        mat = hamiltonian_matrix(dets, ints.h_active_effective(),
                                 ints.active_eri()).toarray()
        w, v = np.linalg.eigh(mat)
        g1, g2 = rdm12(dets, v[:, 0])
        return w[0] + ints.e_core, g1, g2
    return ci_solver


def test_extra_columns_put_a_solver_number_in_the_shared_table(system, kuiva_caplog):
    """⚠ The seam that made the DMRG truncation weight visible without teaching the shared
    driver what a bond dimension is.

    ``max_discarded`` is a tensor-network quality number: it is computed every sweep, and the
    sweep table runs at DEBUG inside a CASSCF (one table per sweep times many
    macro-iterations is noise in a file that *is* the output), so it appeared nowhere at INFO
    at all. A production DMRG run never showed how much of the state it had thrown away.

    The fix is an **additive keyword**, not a restructuring of the validated loop: the caller
    supplies ``(Column, zero-argument getter)`` pairs and the driver appends them. This test
    is the mechanism — the value reaches the row, once per accepted iteration, from a getter
    the driver knows nothing about.
    """
    from kuiva.util import output as out

    factors, h_ao, c0, spaces, nelec = system
    seen = []

    def getter():
        seen.append(len(seen))
        return 10.0 ** -(3 + len(seen))

    kuiva_caplog.set_level("INFO")
    result = optimize_orbitals(factors, h_ao, c0, spaces, _exact_ci(spaces, nelec),
                               max_iter=3, e_nuc=0.0, mode="second-order", report=True,
                               extra_columns=((out.col_sci("w_disc"), getter),))

    assert len(seen) == result.n_iterations                # one call per reported row
    text = "\n".join(r.message for r in kuiva_caplog.records)
    assert "w_disc" in text                                # the header carries the column
    assert "1.000e-04" in text                             # ...and the first row its value


def test_the_table_is_unchanged_when_no_extra_columns_are_given(system, kuiva_caplog):
    """The default is empty, so every existing output — and every committed reference — is
    byte-for-byte what it was. An additive keyword that changed the default would not be one.
    """
    factors, h_ao, c0, spaces, nelec = system
    kuiva_caplog.set_level("INFO")
    optimize_orbitals(factors, h_ao, c0, spaces, _exact_ci(spaces, nelec),
                      max_iter=2, e_nuc=0.0, mode="second-order", report=True)
    text = "\n".join(r.message for r in kuiva_caplog.records)
    assert "w_disc" not in text
    header = [line for line in text.splitlines() if "max rot" in line]
    assert header and header[0].rstrip().endswith("wall [s]")
