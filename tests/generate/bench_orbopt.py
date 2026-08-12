"""Fast orbital-optimizer convergence benchmark.

Why this exists
---------------
The optimizer's behaviour has to be re-measured after every change to it, and the system that
originally exposed the interesting behaviour — TiCl3, 11864 complex rotation parameters — takes
15 to 90 minutes per comparison. That is not a development benchmark, it is an afternoon. This
module is the small proxy: **Ti(2+), 84 spinors, ~1700 parameters**, which keeps what makes the
problem hard and drops what only makes it slow.

What is kept, and why it is the part that matters:

* a **near-degenerate open d shell** — the source of the ill-conditioned Hessian (curvature
  spread of order 10^3), which is what separates the optimizer modes;
* **real relativistic integrals, complex spinors and spin-orbit coupling** — the actual
  arithmetic, not a synthetic Hamiltonian with no structure;
* a **large secondary space** relative to the active one, so the parameter count is dominated
  by virtual-inactive rotations exactly as in the real system.

What is dropped is size alone: TiCl3's cost is driven by ``nao`` (its 126 virtual spinors), not
by its active space, so a bare ion in the same basis is the same problem an order of magnitude
smaller.

The four experiments are deliberately **separate**, because one run that mixes them confounds
them:

* **A — optimizer difficulty.** CAS(2,10) over the 3d shell solved by *exact* CI, so the energy
  surface is smooth and deterministic and the only variable is the step type. It is also the
  regression guard for the event-gated driver: with a static solver no event can ever adopt,
  so any difference from :func:`~kuiva.mcscf.orbopt.optimize_orbitals` is pure overhead.
* **B — surface smoothness, real CI.** CAS(10,18) with the selected cheap CI under three space
  policies: re-selecting every macro-iteration, frozen after the first solve, and event-gated
  (:mod:`kuiva.mcscf.events`). This is the experiment the design rests on.
* **C — the amplitude dial.** The same ion, exact CI in a *truncated* determinant subspace
  chosen perturbatively. The retained count is a knob on the surface-to-surface jump amplitude
  that experiment B does not have — the design record found that ``max_determinants``
  is *not* such a knob — so the controller can be tested against noise of a known size.
* **D — the DMRG cost model.** Experiment C re-accounted: for a DMRG a proposal costs a full
  solve, so the metric is total solver calls rather than macro-iterations, and the question is
  what the event cadence and its backoff cost against what they buy.

Every run is bounded by an **iteration budget and a wall budget enforced from inside** (via the
optimizer's callback), and every row is written the moment it completes. A run that hits its
budget still yields data; the bounded-run rule exists because one that did not, did not.
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from progress import DEFAULT_BUDGET, Heartbeat                      # noqa: E402

from kuiva.ci.strings import (Determinants, connections, diagonal_energies,  # noqa: E402
                              hamiltonian_matrix, rdm12)
from kuiva.interface.api import Molecule, spinor_reference          # noqa: E402
from kuiva.mcscf.adaptive import Proposal, array_key                # noqa: E402
from kuiva.mcscf.events import optimize_orbitals_events             # noqa: E402
from kuiva.mcscf.orbopt import OrbitalSpaces, optimize_orbitals     # noqa: E402
from kuiva.mcscf.preopt import CheapCISolver, cheap_ci              # noqa: E402
from kuiva.util.logging import set_verbosity                        # noqa: E402

#: The proxy system. Ti(2+) is d2: an open, near-degenerate 3d shell in a small basis.
SYSTEM = dict(symbol="Ti", charge=2, spin=2, basis="x2c-SVPall-2c")


def build(n_active_spinor: int, n_active_elec: int):
    """Scalar reference plus orbital spaces for the proxy system."""
    set_verbosity("ERROR")
    mol = Molecule([(SYSTEM["symbol"], (0.0, 0.0, 0.0))], basis=SYSTEM["basis"],
                   charge=SYSTEM["charge"], spin=SYSTEM["spin"])
    ref = spinor_reference(mol)
    # ⚠ From the electron count, not from `2 * (mo_occ > 0).sum()`, which over-counts by one
    # per open shell (an ROHF singly occupied MO has occ > 0 and holds one electron) and can
    # leave an odd inactive count, splitting a Kramers pair across the boundary.
    n_inactive = ref.data.nelec_total - n_active_elec
    spaces = OrbitalSpaces.from_counts(n_inactive, n_active_spinor, ref.nspinor)
    return ref, spaces


def exact_ci_solver(spaces: OrbitalSpaces, n_elec: int, n_states: int = 1):
    """Exact CI in the active space — deterministic, hence a smooth energy surface."""
    dets = Determinants.from_occupations(
        itertools.combinations(range(spaces.n_active), n_elec), spaces.n_active)
    conn = connections(dets)
    weights = np.full(n_states, 1.0 / n_states)

    def solve(ints):
        mat = hamiltonian_matrix(dets, ints.h_active_effective(), ints.active_eri(), conn)
        w, v = np.linalg.eigh(mat.toarray())
        g1, g2 = rdm12(dets, v[:, :n_states], weights, conn)
        return float(np.dot(weights, w[:n_states])) + ints.e_core, g1, g2

    return solve, dets.ndet


def reselecting_solver(n_elec: int, n_states: int, max_determinants: int, **ci_kwargs):
    """The stock cheap CI called afresh at every point — today's behaviour, and the problem.

    Its determinant space is re-chosen for every set of integrals, so ``E(kappa)`` is a family
    of surfaces rather than one, which is what the event-gated controller exists to fix. Kept
    here as the baseline the fix is measured against.
    """
    weights = np.full(n_states, 1.0 / n_states)

    def solve(ints):
        res = cheap_ci(ints.h_active_effective(), ints.active_eri(), n_elec,
                       n_states=n_states, max_determinants=max_determinants, **ci_kwargs)
        return float(np.dot(res.weights, res.energies)) + ints.e_core, res.gamma, res.gamma2

    return solve


# --- C/D: exact CI in a truncated subspace, with the truncation as the noise dial ------------

class TruncationSolver:
    """Exact CI restricted to the ``n_keep`` perturbatively most important determinants.

    The chart family experiment C needs, with an **amplitude knob experiment B does not
    have**. A chart is a subset of the full determinant space; ``solve`` diagonalizes the
    Hamiltonian restricted to the incumbent subset, so the energy is variational, the RDMs are
    exactly those of the returned wavefunction, and the orbital gradient built from them is
    therefore the *exact* gradient of the energy this solver reports. That consistency is why
    the truncation is done this way rather than by perturbing the RDMs, which was the original
    sketch: a perturbed RDM and an unperturbed energy disagree, and the optimizer would then
    misbehave for a reason that has nothing to do with charts.

    ``propose`` re-ranks the *whole* space by first-order perturbation theory at the current
    point and keeps the top ``n_keep`` — a genuine argmax selection, so it has genuine decision
    boundaries, and the trajectory crossing one is exactly the knife-edge flip measured on the
    real system.

    ``n_keep`` is the dial: at ``n_keep = ndet`` there is no truncation and no jump at all; the
    smaller it is, the larger the surface-to-surface gap. ``n_states`` averages, because
    state averaging is where the real selection inconsistency lives (the stock ranking is
    against the ground root while the objective averages several).
    """

    def __init__(self, n_active: int, n_elec: int, n_keep: int, n_states: int = 1):
        self.full = Determinants.from_occupations(
            itertools.combinations(range(n_active), n_elec), n_active)
        self.full_conn = connections(self.full)
        self.n_keep = min(int(n_keep), self.full.ndet)
        self.weights = np.full(n_states, 1.0 / n_states)
        self.n_states = int(n_states)
        self._idx = None                        # indices into `full` making up the incumbent
        self._dets = None
        self._conn = None
        self._key = None
        self._candidate = None
        self.n_solves = 0
        self.n_selections = 0

    # -- the AdaptiveCISolver contract ---------------------------------------------------
    def solve(self, ints):
        h, eri = ints.h_active_effective(), ints.active_eri()
        if self._idx is None:
            self._install(self._rank(h, eri, None))
        e, g1, g2 = self._solve_subset(self._dets, self._conn, h, eri)
        self.n_solves += 1
        return e + ints.e_core, g1, g2

    def propose(self, ints):
        h, eri = ints.h_active_effective(), ints.active_eri()
        self.n_selections += 1
        idx = self._rank(h, eri, None if self._idx is None
                         else self._incumbent_vector(h, eri))
        key = array_key(self.full.masks[idx])
        if key == self._key:
            return None
        dets, conn = self._subset(idx)
        e, g1, g2 = self._solve_subset(dets, conn, h, eri)
        self._candidate = (key, idx)
        shared = 0 if self._idx is None else int(np.intersect1d(idx, self._idx).size)
        return Proposal(energy=e + ints.e_core, gamma=g1, gamma2=g2, key=key,
                        label="{}/{} shared".format(shared, idx.size))

    def adopt(self, key):
        if self._candidate is None or self._candidate[0] != key:
            raise ValueError("no proposal with key {!r} is pending".format(key))
        self._install(self._candidate[1])

    def space_key(self):
        return self._key

    # -- internals -----------------------------------------------------------------------
    def _subset(self, idx):
        dets = Determinants(masks=self.full.masks[idx], n_spinor=self.full.n_spinor,
                            n_elec=self.full.n_elec)
        return dets, connections(dets)

    def _install(self, idx):
        self._idx = np.sort(idx)
        self._dets, self._conn = self._subset(self._idx)
        self._key = array_key(self._dets.masks)
        self._candidate = None

    def _solve_subset(self, dets, conn, h, eri):
        mat = hamiltonian_matrix(dets, h, eri, conn).toarray()
        w, v = np.linalg.eigh(mat)
        k = min(self.n_states, dets.ndet)
        g1, g2 = rdm12(dets, np.asarray(v[:, :k], dtype=complex), self.weights[:k], conn)
        return float(np.dot(self.weights[:k], w[:k])), g1, g2

    def _incumbent_vector(self, h, eri):
        """The incumbent wavefunction, embedded in the full space (zero outside the subset)."""
        _, v = np.linalg.eigh(hamiltonian_matrix(self._dets, h, eri, self._conn).toarray())
        full = np.zeros((self.full.ndet, self.n_states), dtype=complex)
        full[self._idx] = v[:, :self.n_states]
        return full

    def _rank(self, h, eri, vec):
        """Top ``n_keep`` determinants by estimated squared coefficient (ASCI's criterion).

        With no incumbent (the first call) the ranking is by diagonal energy, the cheapest
        sensible starting subset. Afterwards every determinant of the full space is scored by
        the *same* quantity — its weight in the wavefunction, ``|c_D|^2`` — taken exactly
        inside the incumbent subspace and from first-order perturbation theory,
        ``|<D|H|psi> / (E - H_DD)|^2``, outside it. Scoring both on one scale is what lets the
        subset genuinely turn over: a retained determinant whose coefficient has collapsed can
        be displaced by an outside one whose estimate has grown.

        ⚠ Ranking the *whole* space rather than a candidate pool is deliberate, and is where
        this differs from ``preopt._rank_candidates``: it removes the generator truncation as
        a confounder, leaving the decision boundary of the argmax itself, which is the thing
        under test. It is affordable only because this space is small.
        """
        diag = np.real(diagonal_energies(self.full, h, eri))
        if vec is None:
            return np.argsort(diag)[:self.n_keep]
        mat = hamiltonian_matrix(self.full, h, eri, self.full_conn)
        amp = mat @ vec                                          # (ndet, nstate)
        e0 = np.real(np.einsum("dk,dk->k", vec.conj(), amp))
        denom = diag[:, None] - e0[None, :]
        denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
        c1 = np.abs(amp / denom) ** 2                            # first-order coefficient^2
        inside = np.zeros(self.full.ndet, dtype=bool)
        inside[self._idx] = True
        c1[inside] = np.abs(vec[inside]) ** 2                    # exact where we have it
        return np.argsort(-(c1 @ self.weights))[:self.n_keep]


# --- runners --------------------------------------------------------------------------------

def _row(name, label, res, t0, extra=""):
    return ("%-22s %-24s iter=%3d hvp=%5d work=%7.0f |g|=%9.2e wall=%6.1fs E=%.8f%s%s"
            % (name, label, res.n_iterations, res.n_hessian_matvec, res.work_units,
               res.grad_norm, time.time() - t0, res.energy, extra,
               "" if res.converged else "  NOTCONV"))


def _emit(row, out):
    print(row, flush=True)
    with open(out, "a") as fh:                       # incremental: a killed run keeps its rows
        fh.write(row + "\n")


def run_one(name, ref, spaces, solver, mode, max_iter, budget, conv_grad, out):
    """One plain-driver run, heartbeating and self-terminating on the wall budget."""
    hb = Heartbeat(name, budget_seconds=budget,
                   meta={"mode": mode, "n_param": int(spaces.rotation_pairs()[0].size)})

    def cb(info):
        return hb.tick(info["iteration"], grad=info["grad_norm"], energy=info["energy"],
                       hvp=info["n_hessian_matvec"])

    t0 = time.time()
    res = optimize_orbitals(ref.factors, ref.h_one_electron(), ref.spinors_in_ao(), spaces,
                            solver, e_nuc=ref.data.e_nuc, mode=mode, max_iter=max_iter,
                            conv_grad=conv_grad, report=False, callback=cb)
    hb.finish(grad=res.grad_norm, energy=res.energy)
    _emit(_row(name, mode, res, t0), out)
    return res


def run_event(name, ref, spaces, solver, mode, max_iter, budget, conv_grad, out, *,
              label=None, **event_kwargs):
    """One event-gated run. The extra columns are the space trajectory, which is the point."""
    hb = Heartbeat(name, budget_seconds=budget,
                   meta={"mode": mode, "n_param": int(spaces.rotation_pairs()[0].size)})

    def cb(info):
        return hb.tick(info["iteration"], grad=info["grad_norm"], energy=info["energy"],
                       hvp=info["n_hessian_matvec"], adoptions=info["n_adoptions"])

    t0 = time.time()
    res = optimize_orbitals_events(ref.factors, ref.h_one_electron(), ref.spinors_in_ao(),
                                   spaces, solver, e_nuc=ref.data.e_nuc, mode=mode,
                                   max_iter=max_iter, conv_grad=conv_grad, report=False,
                                   callback=cb, **event_kwargs)
    hb.finish(grad=res.grad_norm, energy=res.energy)
    extra = "  ev=%d/%d" % (res.n_adoptions, res.n_events)
    if not res.event_stable and res.converged:
        extra += " UNSTABLE"
    calls = (getattr(solver, "n_solves", 0), getattr(solver, "n_selections", 0))
    if any(calls):
        extra += "  solves=%d sel=%d" % calls
    _emit(_row(name, label or mode, res, t0, extra), out)
    return res


# --- experiments ------------------------------------------------------------------------------

def test_a(args, out):
    """Smooth surface: the step engines, plus the event driver's overhead on a static solver."""
    ref, spaces = build(n_active_spinor=10, n_active_elec=2)
    solver, ndet = exact_ci_solver(spaces, 2)
    print("A: Ti(2+) CAS(2,10) exact CI, %d determinants, %d complex parameters"
          % (ndet, spaces.rotation_pairs()[0].size), flush=True)
    for mode in args.modes:
        run_one("A-exact", ref, spaces, solver, mode, args.max_iter,
                args.budget, args.conv_grad, out)
        # The regression guard: a plain callable through the event driver can never adopt, so
        # a difference here is the controller's overhead and nothing else. ⚠ Give both rows a
        # wall budget they will not hit, or the comparison measures the machine's temperature
        # rather than the driver.
        run_event("A-exact-event", ref, spaces, solver, mode, args.max_iter,
                  args.budget, args.conv_grad, out,
                  label="{} (event driver)".format(mode))


def test_b(args, out):
    """Real selected CI: re-selecting vs frozen vs event-gated, plus the knob sweeps."""
    ref, spaces = build(n_active_spinor=18, n_active_elec=10)
    print("B: Ti(2+) CAS(10,18) selected CI, %d complex parameters"
          % spaces.rotation_pairs()[0].size, flush=True)
    nd, ns = args.max_determinants, args.n_states
    common = dict(max_iter=args.max_iter, budget=args.budget, conv_grad=args.conv_grad)

    for policy in args.policies:
        if policy == "reselect":
            for mode in args.modes:
                run_one("B-reselect", ref, spaces,
                        reselecting_solver(10, ns, nd), mode, out=out, **common)
        elif policy == "frozen":
            for mode in args.modes:
                # `CheapCISolver.solve` selects once and then holds the space: exactly the
                # `freeze_determinants=True` behaviour, through the new implementation.
                run_one("B-frozen", ref, spaces,
                        CheapCISolver(10, n_states=ns, max_determinants=nd).solve,
                        mode, out=out, **common)
        elif policy == "event":
            for mode in args.modes:
                for tau in args.taus:
                    for interval in args.intervals:
                        for keep in args.keep_memory:
                            for ens in args.ensemble:
                                s = CheapCISolver(10, n_states=ns, max_determinants=nd,
                                                  ensemble_selection=ens)
                                label = "%s tau=%.0e ev=%d%s%s" % (
                                    mode, tau, interval, " keep" if keep else "",
                                    " ens" if ens else "")
                                run_event("B-event", ref, spaces, s, mode, out=out,
                                          label=label, tau=tau, event_interval=interval,
                                          max_event_interval=args.max_event_interval,
                                          keep_memory_on_adopt=keep, **common)


def _truncation_rows(args, out, name, cost_note=""):
    ref, spaces = build(n_active_spinor=args.c_active, n_active_elec=args.c_elec)
    probe = TruncationSolver(args.c_active, args.c_elec, args.c_keep, args.n_states)
    print("%s: Ti(2+) CAS(%d,%d) exact CI, %d determinants truncated to %d%s"
          % (name, args.c_elec, args.c_active, probe.full.ndet, probe.n_keep, cost_note),
          flush=True)
    common = dict(max_iter=args.max_iter, budget=args.budget, conv_grad=args.conv_grad)

    def fresh():
        return TruncationSolver(args.c_active, args.c_elec, args.c_keep, args.n_states)

    for mode in args.modes:
        if "reselect" in args.policies:
            # A solver that adopts its own proposal every call: the discontinuous baseline,
            # built from the same chart family so the only difference is who owns the space.
            s = fresh()

            def reselect(ints, _s=s):
                p = _s.propose(ints)
                if p is not None:
                    _s.adopt(p.key)
                return _s.solve(ints)

            run_one(name + "-reselect", ref, spaces, reselect, mode, out=out, **common)
        if "frozen" in args.policies:
            run_one(name + "-frozen", ref, spaces, fresh().solve, mode, out=out, **common)
        if "event" in args.policies:
            for tau in args.taus:
                for interval in args.intervals:
                    for keep in args.keep_memory:
                        s = fresh()
                        label = "%s tau=%.0e ev=%d%s" % (mode, tau, interval,
                                                         " keep" if keep else "")
                        run_event(name + "-event", ref, spaces, s, mode, out=out, label=label,
                                  tau=tau, event_interval=interval,
                                  max_event_interval=args.max_event_interval,
                                  keep_memory_on_adopt=keep, **common)


def test_c(args, out):
    """The amplitude dial: controlled truncation noise around a smooth exact CI."""
    _truncation_rows(args, out, "C")


def test_d(args, out):
    """The DMRG cost model: a proposal costs a full solve, so count solver calls."""
    _truncation_rows(args, out, "D", cost_note="; cost = solves + selections")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test", choices=["A", "B", "C", "D", "both", "all"], default="both")
    p.add_argument("--max-iter", type=int, default=60)
    p.add_argument("--budget", type=float, default=DEFAULT_BUDGET,
                   help="wall budget per run [s]; the ten-minute rule caps ad-hoc runs")
    p.add_argument("--conv-grad", type=float, default=1e-5)
    p.add_argument("--out", default="/tmp/kuiva-runs/bench_orbopt.log")
    p.add_argument("--append", action="store_true", help="keep an existing log")
    # -- what to run, and the knobs Stage 3 of the event-gating design sweeps ------------
    p.add_argument("--policies", default="reselect,frozen,event",
                   help="space policies for tests B/C/D")
    p.add_argument("--modes", default="second-order", help="inner step engines")
    p.add_argument("--taus", default="1e-6", help="adoption thresholds [Eh]")
    p.add_argument("--intervals", default="1", help="initial event intervals")
    p.add_argument("--max-event-interval", type=int, default=16)
    p.add_argument("--keep-memory", default="0",
                   help="0 = reset curvature memory on adoption, 1 = transport it")
    p.add_argument("--ensemble", default="0",
                   help="0/1: rank the cheap-CI selection against the averaged ensemble")
    p.add_argument("--n-states", type=int, default=2)
    p.add_argument("--max-determinants", type=int, default=2000)
    p.add_argument("--c-active", type=int, default=12, help="test C/D active spinors")
    p.add_argument("--c-elec", type=int, default=4, help="test C/D active electrons")
    p.add_argument("--c-keep", type=int, default=120,
                   help="test C/D retained determinants — the noise-amplitude dial")
    args = p.parse_args(argv)

    args.policies = args.policies.split(",")
    args.modes = args.modes.split(",")
    args.taus = [float(x) for x in args.taus.split(",")]
    args.intervals = [int(x) for x in args.intervals.split(",")]
    args.keep_memory = [bool(int(x)) for x in args.keep_memory.split(",")]
    args.ensemble = [bool(int(x)) for x in args.ensemble.split(",")]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if not args.append:
        Path(args.out).write_text("")

    run = {"A": [test_a], "B": [test_b], "C": [test_c], "D": [test_d],
           "both": [test_a, test_b], "all": [test_a, test_b, test_c, test_d]}[args.test]
    for fn in run:
        fn(args, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
