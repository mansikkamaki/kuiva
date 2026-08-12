"""The two-step wall for DMRG-CASSCF, finally measured (the recorded two-step-wall question).

The adaptive design closed with: "This does not touch the two-step wall. Event gating recovers
*frozen-chart* quality on an adaptive surface; it does not exceed it. The DMRG case
remains a hypothesis until the network solver can test it." It now exists; this script makes the
measurement, on the same three-policy design example 9 used for the selected CI:

* ``exact``        — ``FullCISolver``: the wall itself (what a frozen chart can reach);
* ``frozen``       — ``DMRGSolver`` at a **binding** bond-dimension cap, fixed topology;
* ``event``        — the same solver, adaptive (entropy acceptance rule), driven by
  ``optimize_orbitals_events``: proposals gated variationally at fixed integrals;
* ``event-weight`` — the same, with the ``weight`` acceptance rule — O1's "direct error
  proxy for capped runs", which this measurement promoted to the solver's default;
* ``re-adapt``     — the anti-pattern: a wrapper whose ``solve`` re-runs the adaptive
  reconnection at every call, i.e. a solver that hops charts inside the trust region.

The cap must genuinely bind (``max_bond`` below the active space's Schmidt rank) or every
chart represents every state exactly and the comparison is vacuous — the reconnection lesson
about tests too small to force topology, applied here deliberately.

Usage (bounded, incremental, heartbeat)::

    python tests/generate/dmrg_wall.py                 # ~1-2 CPU min, JSON to stdout dir

Results land in ``temp/dmrg_wall.json`` (the git-ignored scratch of the generators) and,
once read, in the local validation notes.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_BLOCKTIME", "0")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import numpy as np                                     # noqa: E402

from kuiva.dmrg import DMRGSolver                       # noqa: E402
from kuiva.dmrg.reconnect import ReconnectionPolicy     # noqa: E402
from kuiva.integrals.transform import ThreeIndexAO      # noqa: E402
from kuiva.mcscf.casci import FullCISolver              # noqa: E402
from kuiva.mcscf.events import optimize_orbitals_events  # noqa: E402
from kuiva.mcscf.orbopt import OrbitalSpaces, optimize_orbitals  # noqa: E402
from progress import Heartbeat                          # noqa: E402

OUT = REPO / "temp"
MAX_ITER = 60
CONV_GRAD = 1e-4
N_ACT, N_ELEC = 6, 2
MAX_BOND = 3                     # binding: the 6-mode half-space Schmidt rank is 8


def system(seed=3, nao=8):
    """A synthetic two-component system with a 6-spinor active space (test conventions)."""
    rng = np.random.default_rng(seed)
    n = 2 * nao
    npair = nao * (nao + 1) // 2
    factors = ThreeIndexAO(l_packed=rng.standard_normal((3 * nao, npair)), nao=nao,
                           origin="cholesky")
    h_ao = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    h_ao = 0.5 * (h_ao + h_ao.conj().T)
    _, c0 = np.linalg.eigh(h_ao)
    return factors, h_ao, np.ascontiguousarray(c0), \
        OrbitalSpaces.from_counts(2, N_ACT, n)


class ReAdaptingSolver:
    """The re-selecting anti-pattern, DMRG flavour: ``solve`` re-adapts the topology every call.

    Wrapping this in ``StaticSolver`` would gate nothing (the mcscf.adaptive warning);
    it exists here to measure what the contract's ``solve``-holds-its-space rule buys.
    """

    def __init__(self, **kw):
        self._inner = DMRGSolver(adaptive=True, **kw)

    def __call__(self, ints):
        e, g, g2 = self._inner.solve(ints)
        proposal = self._inner.propose(ints)
        if proposal is not None:
            self._inner.adopt(proposal.key)             # unconditional: no gate, no margin
            return self._inner.solve(ints)
        return e, g, g2


def run_case(name, factors, h_ao, c0, spaces, hb):
    t_wall, t_cpu = time.time(), time.process_time()
    common = dict(max_iter=MAX_ITER, mode="second-order", conv_grad=CONV_GRAD,
                  report=False)
    dm_kw = dict(n_elec=N_ELEC, max_bond=MAX_BOND, n_roots=1, enforce_kramers=False,
                 seed=2, max_sweeps=20)
    solver = None
    if name == "exact":
        solver = FullCISolver(N_ACT, N_ELEC, n_states=1, enforce_kramers=False)
        result = optimize_orbitals(factors, h_ao, c0, spaces, solver, **common)
        extra = {}
    elif name == "frozen":
        solver = DMRGSolver(adaptive=False, **dm_kw)
        result = optimize_orbitals(factors, h_ao, c0, spaces, solver, **common)
        extra = {"solves": solver.n_solves}
    elif name.startswith("event"):
        rule = "weight" if name.endswith("weight") else "entropy"
        solver = DMRGSolver(adaptive=True, policy=ReconnectionPolicy(rule=rule),
                            **dm_kw)
        result = optimize_orbitals_events(factors, h_ao, c0, spaces, solver, **common)
        extra = {"solves": solver.n_solves, "proposals": solver.n_proposals,
                 "adoptions": solver.n_adoptions,
                 "event_stable": bool(result.event_stable)}
    elif name == "re-adapt":
        wrapper = ReAdaptingSolver(**dm_kw)
        result = optimize_orbitals(factors, h_ao, c0, spaces, wrapper, **common)
        solver = wrapper._inner
        extra = {"solves": solver.n_solves, "adoptions": solver.n_adoptions}
    rec = {"case": name, "energy": float(result.energy),
           "grad_norm": float(result.grad_norm), "converged": bool(result.converged),
           "iterations": int(result.n_iterations),
           "wall_s": time.time() - t_wall, "cpu_s": time.process_time() - t_cpu}
    rec.update(extra)
    hb.tick(0, **rec)
    print(json.dumps(rec))
    return rec


def main():
    OUT.mkdir(exist_ok=True)
    hb = Heartbeat("dmrg_wall", meta={"max_bond": MAX_BOND, "cas": [N_ELEC, N_ACT]})
    factors, h_ao, c0, spaces = system()
    records = []
    for name in ("exact", "frozen", "event", "event-weight", "re-adapt"):
        records.append(run_case(name, factors, h_ao, c0, spaces, hb))
        (OUT / "dmrg_wall.json").write_text(json.dumps(records, indent=1))
        if hb.expired:
            print("budget spent; stopping after", name)
            break
    exact = records[0]
    for rec in records[1:]:
        rec["e_above_exact"] = rec["energy"] - exact["energy"]
    (OUT / "dmrg_wall.json").write_text(json.dumps(records, indent=1))
    hb.finish(cases=len(records))
    print(json.dumps(records, indent=1))


if __name__ == "__main__":
    main()
