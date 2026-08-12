"""Tier-0 tests for the adaptive-solver contract and the event-gated optimizer.

Two things are under test and they are deliberately separated.

**The contract** (:mod:`kuiva.mcscf.adaptive`, :class:`~kuiva.mcscf.preopt.CheapCISolver`):
that a plain callable still behaves exactly as it does today, that a solver holds its space
between events, and that a proposal changes nothing until it is adopted. Plus the robustness
fix underneath it — a Lanczos solve that fails must fall back, not crash.

**The controller** (:mod:`kuiva.mcscf.events`): driven by a *scripted* solver over exact CI in
a sequence of nested determinant subsets. No chemistry is needed and none is used: nested
subsets make each proposed chart variationally better by construction, which is exactly the
structure the controller's guarantees are stated over — monotone energy across adoptions,
accept/reject on one surface at a time, curvature memory scoped to the chart, termination on
event stability rather than on a gradient norm alone, and a failed solve rejecting a step
instead of ending the run.

⚠ The scripted charts are a *mechanism* test. Whether event gating wins on a real surface is
a benchmark question and is measured elsewhere; the recorded dead-end table is the standing
warning against tuning a controller on synthetic evidence.
"""
import itertools

import numpy as np
import pytest

from kuiva.ci.strings import Determinants, hamiltonian_matrix, rdm12
from kuiva.integrals.transform import ThreeIndexAO
from kuiva.mcscf import preopt
from kuiva.mcscf.adaptive import (AdaptiveCISolver, Proposal, SolverFailure, StaticSolver,
                                  array_key, as_adaptive_solver)
from kuiva.mcscf.events import optimize_orbitals_events
from kuiva.mcscf.orbopt import (CASIntegrals, OrbitalOptimizer, OrbitalSpaces,
                                optimize_orbitals, unitary_from_antihermitian)
from kuiva.mcscf.preopt import CheapCISolver, cheap_ci


# --- a small synthetic two-component system, as in test_mcscf.py -----------------------------
@pytest.fixture(scope="module")
def system():
    """AO factors, a 2c one-electron Hamiltonian, orthonormal spinors, spaces, electrons.

    Six active spinors and three electrons, so a determinant budget of 10 out of 20 leaves
    the cheap CI something to select — which is what the solver tests need.
    """
    rng = np.random.default_rng(5)
    nao = 6
    n = 2 * nao
    npair = nao * (nao + 1) // 2
    factors = ThreeIndexAO(l_packed=0.5 * rng.standard_normal((3 * nao, npair)), nao=nao,
                           origin="cholesky")
    h_ao = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    h_ao = 0.5 * (h_ao + h_ao.conj().T)
    _, c0 = np.linalg.eigh(h_ao)
    spaces = OrbitalSpaces.from_counts(n_inactive=2, n_active=6, n_orb=n)
    return factors, h_ao, np.ascontiguousarray(c0), spaces, 3


@pytest.fixture(scope="module")
def chart_system():
    """The converging system of ``tests/test_mcscf.py`` — CAS(2,4) from a core guess.

    The controller tests need a surface the *inner* engine actually converges on, so that a
    failure to converge is attributable to the event logic rather than to the underlying
    optimization being hard. This is that system, seed for seed.
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
    return factors, h_ao, np.ascontiguousarray(c0), spaces, 2


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


# --- the adapter: a plain callable must keep behaving exactly as it does today ---------------
def test_static_solver_is_the_callable_and_nothing_else():
    calls = []

    def ci(ints):
        calls.append(ints)
        return (-1.25, "gamma", "gamma2")

    s = as_adaptive_solver(ci)
    assert isinstance(s, StaticSolver)
    assert s.solve("ints") == (-1.25, "gamma", "gamma2")
    assert calls == ["ints"]
    assert s.propose("ints") is None            # no event can ever fire
    assert s.space_key() == s.space_key()       # one surface for the whole run
    with pytest.raises(ValueError, match="nothing to adopt"):
        s.adopt("anything")


def test_a_protocol_implementation_is_passed_through_unwrapped():
    solver = CheapCISolver(2)
    assert isinstance(solver, AdaptiveCISolver)
    assert as_adaptive_solver(solver) is solver


def test_a_non_callable_is_refused():
    with pytest.raises(TypeError, match="callable"):
        as_adaptive_solver(object())


def test_array_key_identifies_the_space_not_its_order():
    a = np.array([7, 2, 9, 4], dtype=np.uint64)
    assert array_key(a) == array_key(a[::-1])            # same space, different order
    assert array_key(a) != array_key(np.array([7, 2, 9, 5], dtype=np.uint64))
    assert array_key(a) == array_key(a)                  # stable within a process
    assert isinstance(array_key(a), str)


def test_the_event_driver_reproduces_the_plain_driver_for_a_plain_callable(system):
    """A wrapped callable must give **bitwise** the trajectory of ``optimize_orbitals``.

    Run with an unreachable gradient threshold so both loops take the same fixed number of
    iterations and no convergence test can paper over a difference.
    """
    factors, h_ao, c0, spaces, nelec = system
    occ = list(itertools.combinations(range(spaces.n_active), nelec))
    dets = Determinants.from_occupations(occ, spaces.n_active)

    def ci(ints):
        mat = hamiltonian_matrix(dets, ints.h_active_effective(), ints.active_eri()).toarray()
        w, v = np.linalg.eigh(mat)
        g1, g2 = rdm12(dets, v[:, 0])
        return w[0] + ints.e_core, g1, g2

    kw = dict(e_nuc=0.0, max_iter=8, conv_grad=1e-12, mode="auto", report=False)
    plain = optimize_orbitals(factors, h_ao, c0, spaces, ci, **kw)
    gated = optimize_orbitals_events(factors, h_ao, c0, spaces, ci, **kw)

    assert gated.energy == plain.energy
    assert np.array_equal(gated.coeff, plain.coeff)
    assert gated.history == plain.history
    assert gated.n_hessian_matvec == plain.n_hessian_matvec
    assert gated.n_rejected == plain.n_rejected
    assert gated.n_iterations == plain.n_iterations
    # Events are attempted; a static solver has nothing to offer, so none can be adopted and
    # the controller has no way to alter the trajectory.
    assert gated.n_adoptions == 0
    assert all(not e.adopted and e.candidate is None for e in gated.events)


# --- the cheap CI as an adaptive solver -------------------------------------------------------
def test_cheap_ci_solver_holds_its_space_across_solves(system):
    """``solve`` selects once and never again — that is the whole point of the contract."""
    factors, h_ao, c0, spaces, nelec = system
    solver = CheapCISolver(nelec, max_determinants=10, selection_rounds=1)

    ints = CASIntegrals.build(factors, h_ao, c0, spaces)
    e0, g0, _ = solver.solve(ints)
    key = solver.space_key()
    assert solver.n_selections == 1

    rng = np.random.default_rng(0)
    kappa = 0.1 * (rng.standard_normal((spaces.n_orb, spaces.n_orb))
                   + 1j * rng.standard_normal((spaces.n_orb, spaces.n_orb)))
    kappa = kappa - kappa.conj().T
    c1 = np.ascontiguousarray(c0 @ unitary_from_antihermitian(kappa))
    e1, _, _ = solver.solve(CASIntegrals.build(factors, h_ao, c1, spaces))

    assert solver.space_key() == key            # the surface did not change under our feet
    assert solver.n_selections == 1             # and nothing was re-selected
    assert solver.n_solves == 2
    assert e1 != e0                             # but the energy did move: it is a surface


def test_a_proposal_changes_nothing_until_it_is_adopted(system):
    factors, h_ao, c0, spaces, nelec = system
    solver = CheapCISolver(nelec, max_determinants=10, selection_rounds=1)
    ints = CASIntegrals.build(factors, h_ao, c0, spaces)
    solver.solve(ints)

    # A different point: the selection made there is generally a different space.
    rng = np.random.default_rng(3)
    kappa = 0.3 * (rng.standard_normal((spaces.n_orb, spaces.n_orb))
                   + 1j * rng.standard_normal((spaces.n_orb, spaces.n_orb)))
    kappa = kappa - kappa.conj().T
    ints2 = CASIntegrals.build(factors, h_ao,
                               np.ascontiguousarray(c0 @ unitary_from_antihermitian(kappa)),
                               spaces)
    solver.solve(ints2)
    key_before = solver.space_key()

    proposal = solver.propose(ints2)
    assert proposal is None or isinstance(proposal, Proposal)
    assert solver.space_key() == key_before, "propose() must not adopt"
    if proposal is None:
        pytest.skip("the fresh selection reproduced the incumbent space at this point")
    assert proposal.key != key_before

    # The proposal is comparable: it is the same integrals, so only the space differs.
    direct = cheap_ci(ints2.h_active_effective(), ints2.active_eri(), nelec,
                      max_determinants=10, selection_rounds=1)
    assert proposal.energy == pytest.approx(float(direct.energies[0]) + ints2.e_core, abs=1e-9)

    with pytest.raises(ValueError, match="no proposal with key"):
        solver.adopt("not-a-key")

    solver.adopt(proposal.key)
    assert solver.space_key() == proposal.key
    e_after, _, _ = solver.solve(ints2)
    assert e_after == pytest.approx(proposal.energy, abs=1e-10)


# --- D5: the eigensolver may fail, and failing must not be fatal -----------------------------
def _big_space(n=12, nelec=6, seed=17):
    h, eri = random_active_integrals(n, seed=seed)
    dets = Determinants.from_occupations(itertools.combinations(range(n), nelec), n)
    assert dets.ndet > preopt.DENSE_SOLVE_MAX_DET       # so the sparse path is taken
    return dets, h, eri


def test_dense_hamiltonian_gb_matches_a_real_array():
    """Sizing is exact and never pads — bounded on both sides."""
    ndet = 700
    nbytes = np.empty((ndet, ndet), dtype=np.complex128).nbytes
    gb = preopt.dense_hamiltonian_gb(ndet)
    assert gb == pytest.approx(nbytes / 1024 ** 3, rel=1e-12)


def test_the_sparse_solve_agrees_with_the_dense_one_at_the_chosen_tolerance():
    """The ARPACK tolerance is looser than machine precision on purpose, but whatever it is
    becomes a **noise floor on E(kappa)** — so it must stay far inside the 1e-8 Eh the
    optimizer's accept/reject test works at, or the fix for one non-smoothness introduces
    another."""
    dets, h, eri = _big_space()
    w_sparse, _ = preopt._solve(dets, h, eri, 2)
    mat = hamiltonian_matrix(dets, h, eri).toarray()
    w_dense = np.linalg.eigvalsh(mat)[:2]
    assert np.max(np.abs(w_sparse - w_dense)) < 1e-10


def test_the_arpack_tolerance_targets_an_absolute_accuracy():
    """ARPACK's ``tol`` is relative to ``|lambda|``, so a flat relative number means something
    different for an active-space energy of 10 Eh and one of 1000 — hence the derivation from
    an absolute target, bounded at both ends."""
    dets, h, eri = _big_space()
    hmat = hamiltonian_matrix(dets, h, eri)
    scale = abs(float(np.min(np.real(hmat.diagonal()))))
    assert scale > 1.0
    tol = preopt._eigsh_tol(hmat)
    assert preopt.EIGSH_MIN_TOL <= tol <= preopt.EIGSH_TARGET_EH
    assert tol * scale == pytest.approx(preopt.EIGSH_TARGET_EH, rel=1e-9)
    assert tol > np.finfo(float).eps, "machine precision is what made ARPACK grind"

    class _Diag:
        def __init__(self, value):
            self.value = value

        def diagonal(self):
            return np.array([self.value])

    # The energy scale is floored at 1 Eh, so a near-zero Hamiltonian cannot inflate the
    # tolerance; a huge one is stopped by the floor rather than asking for 1e-20.
    assert preopt._eigsh_tol(_Diag(-1e-9)) == preopt.EIGSH_TARGET_EH
    assert preopt._eigsh_tol(_Diag(-1e12)) == preopt.EIGSH_MIN_TOL


def test_a_failed_lanczos_solve_falls_back_to_the_dense_one(monkeypatch, kuiva_caplog):
    """ARPACK non-convergence is a solver's problem, not the calculation's."""
    import scipy.sparse.linalg as spla

    def boom(*args, **kwargs):
        raise spla.ArpackNoConvergence("scripted", np.array([]), np.array([[]]))

    monkeypatch.setattr(spla, "eigsh", boom)
    dets, h, eri = _big_space()
    w, v = preopt._solve(dets, h, eri, 2)
    mat = hamiltonian_matrix(dets, h, eri).toarray()
    assert np.max(np.abs(w - np.linalg.eigvalsh(mat)[:2])) < 1e-10
    assert v.shape == (dets.ndet, 2)
    assert any("ARPACK did not converge" in r.getMessage()
               for r in kuiva_caplog.records)


def test_beyond_the_dense_fallback_a_failed_solve_raises_solver_failure(monkeypatch):
    """Past the fallback size there is no exact escape, so the failure is reported as one —
    which is what the controller turns into a rejected step rather than a crash."""
    import scipy.sparse.linalg as spla

    def boom(*args, **kwargs):
        raise spla.ArpackNoConvergence("scripted", np.array([]), np.array([[]]))

    monkeypatch.setattr(spla, "eigsh", boom)
    monkeypatch.setattr(preopt, "DENSE_FALLBACK_MAX_DET", 10)
    dets, h, eri = _big_space()
    with pytest.raises(SolverFailure, match="did not converge"):
        preopt._solve(dets, h, eri, 2)


# --- ensemble-keyed selection (the default) ---------------------------------------------------
def test_ensemble_selection_is_a_no_op_for_a_single_state():
    """With one root the ensemble *is* the ground state, so the option must change nothing.

    ⚠ Asserted **bitwise**, and the implementation takes the single-root branch explicitly to
    make it so. The two criteria are the same quantity computed by different expressions, and
    ``sqrt(|c|^2)`` versus ``|c|`` can disagree in the last bit — enough to flip an argsort tie
    and silently move a one-state space that nothing in this option is supposed to touch.
    """
    h, eri = random_active_integrals(14, seed=3)
    a = cheap_ci(h, eri, 7, n_states=1, max_determinants=300, ensemble_selection=False)
    b = cheap_ci(h, eri, 7, n_states=1, max_determinants=300, ensemble_selection=True)
    assert np.array_equal(np.sort(a.dets.masks), np.sort(b.dets.masks))
    assert a.energies[0] == b.energies[0]


@pytest.mark.parametrize("seed,n_states", [(3, 2), (11, 3), (31, 2)])
def test_ensemble_selection_lowers_the_state_averaged_energy(seed, n_states):
    """The objective is the state average, so a selection keyed on the ground root alone is
    optimizing something else. Ranking against the ensemble must improve the quantity that is
    actually being minimized — which is why it is the default, and why the *other* setting is
    the one that has to be asked for."""
    h, eri = random_active_integrals(14, seed=seed)
    kw = dict(n_states=n_states, max_determinants=300)
    ground = cheap_ci(h, eri, 7, ensemble_selection=False, **kw)
    ensemble = cheap_ci(h, eri, 7, **kw)                       # the default
    assert not np.array_equal(np.sort(ground.dets.masks), np.sort(ensemble.dets.masks))
    assert (np.dot(ensemble.weights, ensemble.energies)
            < np.dot(ground.weights, ground.energies))


# --- chart-scoped curvature memory -------------------------------------------------------------
def test_reset_chart_discards_the_old_surface_and_restores_the_trust_radius():
    spaces = OrbitalSpaces.from_counts(1, 2, 6)
    opt = OrbitalOptimizer(spaces, max_step=0.2, mode="quasi-newton")
    opt._s.append(np.ones(opt.n_parameters, dtype=complex))
    opt._y.append(np.ones(opt.n_parameters, dtype=complex))
    opt._ah_guess = np.ones(opt.n_parameters, dtype=complex)
    opt._gnorm_history.extend([1.0, 1.0, 1.0])
    opt._stalls = 4
    opt.trust = 1e-6                                   # collapsed on the surface just left

    opt.reset_chart(trust_floor=0.05)
    assert opt._s == [] and opt._y == []
    assert opt._ah_guess is None
    assert opt._gnorm_history == []
    assert opt._stalls == 0
    assert opt.trust == pytest.approx(0.05)

    opt.trust = 0.2
    opt.reset_chart(trust_floor=0.05)                  # a floor, never a reduction
    assert opt.trust == pytest.approx(0.2)

    opt._s.append(np.ones(opt.n_parameters, dtype=complex))
    opt._y.append(np.ones(opt.n_parameters, dtype=complex))
    opt.reset_chart(trust_floor=0.05, keep_memory=True)
    assert len(opt._s) == 1                            # transported, deliberately
    assert opt._prev_grad is None and opt._prev_step is None


# --- the controller, on scripted charts ---------------------------------------------------------
class ScriptedSolver:
    """Exact CI over a scripted sequence of **nested** determinant subsets.

    Each chart is a smooth, deterministic surface with an exact orbital gradient (the RDMs
    are those of a variationally optimized CI in a fixed space), and nesting makes every
    proposal variationally better than its predecessor at any point — so "adopting lowers the
    energy" is a property of the construction, not something the test has to arrange.

    Once the schedule is exhausted, ``propose`` offers the *first* (worst) chart, which must
    be refused: that is what drives the termination test.
    """

    def __init__(self, n_active, nelec, sizes, fail_on_solve=()):
        self.all_dets = Determinants.from_occupations(
            itertools.combinations(range(n_active), nelec), n_active)
        self.sizes = list(sizes)
        self.charts = [Determinants(masks=self.all_dets.masks[:k].copy(),
                                    n_spinor=n_active, n_elec=nelec) for k in self.sizes]
        self.stage = 0
        self._dets = None
        self._candidate = None
        self.fail_on_solve = set(fail_on_solve)
        self.n_solve = 0
        self.n_propose = 0
        self.keys_seen = []

    # -- the contract --
    def solve(self, ints):
        self.n_solve += 1
        if self.n_solve in self.fail_on_solve:
            raise SolverFailure("scripted failure at solve #{}".format(self.n_solve))
        if self._dets is None:
            self._install(self.charts[0])
        self.keys_seen.append(self.space_key())
        return self._exact(self._dets, ints)

    def propose(self, ints):
        self.n_propose += 1
        nxt = self.charts[min(self.stage + 1, len(self.charts) - 1)]
        if self.stage + 1 >= len(self.charts):
            nxt = self.charts[0]                       # exhausted: offer something worse
        if array_key(nxt.masks) == self.space_key():
            return None
        energy, gamma, gamma2 = self._exact(nxt, ints)
        self._candidate = (array_key(nxt.masks), nxt, gamma, gamma2)
        return Proposal(energy=energy, gamma=gamma, gamma2=gamma2,
                        key=self._candidate[0], label="{} dets".format(nxt.ndet))

    def adopt(self, key):
        assert self._candidate is not None and self._candidate[0] == key
        dets = self._candidate[1]
        self._install(dets)
        self.stage = self.sizes.index(dets.ndet)

    def space_key(self):
        return None if self._dets is None else array_key(self._dets.masks)

    # -- internals --
    def _install(self, dets):
        self._dets = dets
        self._candidate = None

    @staticmethod
    def _exact(dets, ints):
        mat = hamiltonian_matrix(dets, ints.h_active_effective(), ints.active_eri()).toarray()
        w, v = np.linalg.eigh(mat)
        g1, g2 = rdm12(dets, v[:, 0])
        return float(w[0]) + ints.e_core, g1, g2


#: The full space is C(4, 2) = 6 determinants, so [2, 4, 6] is a nested ladder ending at the
#: exact CI. ``FULL`` alone is a solver with nothing to propose.
LADDER = [2, 4, 6]
FULL = [6]


def _run(chart_system, solver, **kw):
    factors, h_ao, c0, spaces, _ = chart_system
    opts = dict(e_nuc=0.0, max_iter=40, conv_grad=1e-4, mode="auto", report=False)
    opts.update(kw)
    return optimize_orbitals_events(factors, h_ao, c0, spaces, solver, **opts)


def test_the_energy_is_monotone_across_adoptions(chart_system):
    """Adoption is variational at fixed integrals, so it can only lower the energy — and the
    accepted steps are the usual trust-region descent. The whole trajectory is a descent."""
    _, _, _, spaces, nelec = chart_system
    solver = ScriptedSolver(spaces.n_active, nelec, LADDER)
    res = _run(chart_system, solver)
    assert res.n_adoptions == 2, "both improving charts should have been taken"
    assert np.all(np.diff(np.asarray(res.history)) <= 1e-8), "the energy rose"
    adopted = [r for r in res.events if r.adopted]
    assert len(adopted) == 2
    for rec in adopted:
        assert rec.gain > 0.0 and rec.candidate < rec.incumbent


def test_convergence_is_gated_on_an_event_not_on_the_gradient_alone(chart_system):
    """The honest fixed point of an adaptive-space MCSCF: the orbitals are stationary **and**
    a fresh selection cannot improve them.

    Both halves are asserted. With nothing better available, the run converges — and does so
    only after the mandatory event at the last iteration refused. With something better
    available at the point where the gradient test is met, the run adopts it *instead of*
    declaring convergence, which is precisely the failure mode a gradient-only test has.
    """
    _, _, _, spaces, nelec = chart_system
    stable = ScriptedSolver(spaces.n_active, nelec, FULL)
    res = _run(chart_system, stable, event_interval=1000)     # only mandatory events
    assert res.converged and res.event_stable
    assert res.n_events == 1 and res.n_adoptions == 0
    assert res.events[0].iteration == res.n_iterations        # the terminating event
    assert res.grad_norm < 1e-4

    improving = ScriptedSolver(spaces.n_active, nelec, [5, 6])
    res2 = _run(chart_system, improving, conv_grad=1e-1, event_interval=1000)
    assert res2.n_events == 2
    # The first mandatory event is where a gradient-only test would have stopped the run.
    assert res2.events[0].adopted and res2.events[0].gain > 0.0
    assert res2.events[1].adopted is False
    assert res2.converged and res2.event_stable
    assert improving._dets.ndet == 6


def test_the_incumbent_space_never_changes_inside_the_inner_loop(chart_system):
    """Chart consistency by construction: a trial step is evaluated on the same surface the
    model was built from, so a space change can never be misread as a failed step."""
    _, _, _, spaces, nelec = chart_system
    solver = ScriptedSolver(spaces.n_active, nelec, LADDER)
    res = _run(chart_system, solver)
    changes = sum(1 for a, b in zip(solver.keys_seen, solver.keys_seen[1:]) if a != b)
    assert changes == res.n_adoptions
    assert solver.n_propose == res.n_events


def test_the_curvature_memory_is_reset_on_adoption(monkeypatch, chart_system):
    """Chart scoping: L-BFGS pairs, the augmented-Hessian warm start and the trust radius belong to
    the surface that produced them."""
    import kuiva.mcscf.events as events_mod

    calls = []

    class Spy(OrbitalOptimizer):
        def reset_chart(self, **kw):
            calls.append(dict(kw))
            super().reset_chart(**kw)

    monkeypatch.setattr(events_mod, "OrbitalOptimizer", Spy)
    _, _, _, spaces, nelec = chart_system
    solver = ScriptedSolver(spaces.n_active, nelec, LADDER)
    res = _run(chart_system, solver, max_step=0.2)
    assert len(calls) == res.n_adoptions == 2
    assert all(c["trust_floor"] == pytest.approx(0.05) for c in calls)   # max_step / 4
    assert all(c["keep_memory"] is False for c in calls)


def test_events_back_off_when_proposals_keep_being_refused(chart_system):
    """A proposal costs a full solve, so refusals must make them rare — while the mandatory
    event at convergence keeps the termination guarantee."""
    _, _, _, spaces, nelec = chart_system
    solver = ScriptedSolver(spaces.n_active, nelec, FULL)    # nothing better exists
    res = _run(chart_system, solver, max_iter=30, max_event_interval=8)
    assert res.n_adoptions == 0
    assert res.n_events == res.n_refusals
    # Without backoff this would be one event per accepted iteration.
    assert res.n_events < res.n_iterations / 2, (res.n_events, res.n_iterations)


def test_a_failed_solve_at_a_trial_point_rejects_the_step(chart_system, kuiva_caplog):
    """D5: a point the solver cannot evaluate is a point the optimizer must not move to."""
    _, _, _, spaces, nelec = chart_system
    solver = ScriptedSolver(spaces.n_active, nelec, FULL, fail_on_solve=(2, 3))
    res = _run(chart_system, solver, max_iter=12)
    assert res.n_solver_failures == 2
    assert res.n_rejected >= 2
    assert np.isfinite(res.energy) and res.energy <= res.history[0] + 1e-10
    assert np.all(np.diff(np.asarray(res.history)) <= 1e-8)
    assert any("the CI failed at the trial point" in r.getMessage()
               for r in kuiva_caplog.records)


def test_a_failed_solve_at_the_starting_point_propagates(chart_system):
    """There is nothing to fall back to at iteration zero, and pretending otherwise would
    hand the caller orbitals from a calculation that never ran."""
    _, _, _, spaces, nelec = chart_system
    solver = ScriptedSolver(spaces.n_active, nelec, FULL, fail_on_solve=(1,))
    with pytest.raises(SolverFailure):
        _run(chart_system, solver)


def test_the_result_carries_the_adopted_space(chart_system):
    """The RDMs the caller gets back must come from the adopted space, not the one the run
    left behind — a mismatch there is a density matrix for a surface nobody is on."""
    _, _, _, spaces, nelec = chart_system
    solver = ScriptedSolver(spaces.n_active, nelec, LADDER)
    res = _run(chart_system, solver)
    assert solver._dets.ndet == 6
    assert res.gamma.shape == (spaces.n_active, spaces.n_active)
    assert np.real(np.trace(res.gamma)) == pytest.approx(nelec, abs=1e-8)
    assert res.energy == pytest.approx(res.history[-1], abs=1e-12)
