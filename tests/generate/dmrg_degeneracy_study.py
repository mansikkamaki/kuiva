"""Can a sweep converge degenerate roots to CI quality, and what does it cost?

**What this decides.** The tensor-network validation record reported that a state-averaged
sweep leaves ~1e-9 Eh between the members of a Kramers pair where nothing is truncated,
against 1e-15…1e-13 Eh from the general CI path — root-wise convergence, not broken
time-reversal symmetry. The open question was whether a tighter local convergence criterion
removes it and what that costs. This measures the spread against `conv_tol`, `davidson_tol`
and the bond cap, on both a **saturating** cap (nothing truncated, the record's own case) and
a **truncating** one (where the record says the residual actually lives).

⚠ **Two protocols, because they answer different questions.** `casci` runs ONE sweep at fixed
guess orbitals against an exact CI on the *same* `CASIntegrals`, so the difference is the
sweep and nothing else. `casscf` is the record's own protocol — a full state-averaged
optimization through the class API — and carries the orbital optimizer's own convergence
inside it. A number from the first is a statement about the sweep; a number from the second is
what a user actually sees.

⚠ **The spread is taken inside a degenerate GROUP, not between roots 0 and 1.** For an
odd-electron system with two roots those coincide; for `bi` the ground manifold is 4-fold and
the pairwise spread over the whole block is the quantity. `e_core` is a common shift and
cancels from it either way.

Run::

    python tests/generate/dmrg_degeneracy_study.py --protocol casci
    python tests/generate/dmrg_degeneracy_study.py --protocol casscf --only b

One protocol per invocation stays inside the ten-minute rule; records are written
incrementally after every case.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import systems as sysdef                                                     # noqa: E402

from kuiva.interface import api                                              # noqa: E402
from kuiva.util import resources as res                                      # noqa: E402
from progress import Heartbeat                                               # noqa: E402

WALL_BUDGET_S = 9.5 * 60

#: (system key, active electrons, active spinors, roots, saturating cap, truncating cap).
#: ``b`` is the validation record's own case. ``ce3p`` is 14 spinors, where a cap of 4 is a
#: real truncation of the sweep's Schmidt spectrum. ``bi`` carries a 4-fold degenerate group,
#: so the "solve a degenerate group as a block" half of the question has a case with a block
#: bigger than a Kramers pair.
CASES = (
    ("b",     1,  6,  2, 16, 2),
    ("bi",    3,  6,  4, 16, 4),
    ("ce3p",  1, 14,  6, 32, 4),
)

#: Sweep convergence criteria swept. The solver's defaults are conv_tol 1e-9, davidson 1e-8.
CONV_TOLS = (1e-7, 1e-9, 1e-11, 1e-13)
DAVIDSON_TOLS = (1e-8, 1e-11)


def _system(key: str) -> sysdef.System:
    for s in sysdef.SYSTEMS:
        if s.key == key:
            return s
    raise SystemExit("no system {!r}".format(key))


def _group_spread(energies: np.ndarray, n: int) -> float:
    """Largest gap inside the lowest ``n`` roots — the degenerate group's spread, in Eh."""
    e = np.sort(np.asarray(energies, dtype=float))[:n]
    return float(e[-1] - e[0])


def _reference_and_space(system, n_active: int, n_active_elec: int, *, memory_gb: float):
    molecule = api.Molecule(atoms=system.atoms, basis=system.basis, charge=system.charge,
                            spin=system.spin)
    reference = api.spinor_reference(molecule, memory_gb=memory_gb, screening="none")
    kw = (dict(character=(system.atoms[0][0], system.active_l), n_active=n_active,
               n_active_elec=n_active_elec) if system.active_l
          else dict(active=list(range(reference.data.nelec_total - n_active_elec,
                                      reference.data.nelec_total - n_active_elec + n_active)),
                    n_active_elec=n_active_elec))
    space = api.active_space_for(reference, **kw)
    return reference, space


def _cas_integrals(reference, space):
    from kuiva.mcscf.orbopt import CASIntegrals
    return CASIntegrals.build(reference.factors, reference.h_one_electron(),
                              reference.spinors_in_ao(), space.spaces,
                              e_nuc=reference.data.e_nuc)


def _exact_ci(ints, n_elec: int, n_roots: int) -> Tuple[np.ndarray, float, float]:
    """Exact CI roots on the same integrals — the quality bar the sweep is measured against."""
    from kuiva.mcscf.casci import FullCISolver
    t0, c0 = time.time(), time.process_time()
    solver = FullCISolver(ints.spaces.n_active, n_elec, n_states=n_roots)
    result = solver.casci(ints)
    e = np.asarray(result.total_energies, dtype=float)
    return e, time.time() - t0, time.process_time() - c0


def _sweep(ints, n_elec: int, n_roots: int, *, max_bond: int, conv_tol: float,
           davidson_tol: float) -> Tuple[Optional[np.ndarray], Dict[str, object]]:
    from kuiva.dmrg import DMRGSolver
    from kuiva.dmrg.solver import SolverFailure
    t0, c0 = time.time(), time.process_time()
    solver = DMRGSolver(n_elec, max_bond=max_bond, n_roots=n_roots, conv_tol=conv_tol,
                        davidson_tol=davidson_tol, max_sweeps=60)
    try:
        solver.solve(ints)
    except SolverFailure as exc:
        return None, {"failed": "SolverFailure: {}".format(exc),
                      "wall_s": round(time.time() - t0, 3),
                      "cpu_s": round(time.process_time() - c0, 3)}
    except ValueError as exc:
        # ⚠ A refusal, not a failure: the group-complete truncation rule declines to cut
        # inside a degenerate Schmidt group rather than round it. Recorded as an outcome,
        # because "this cap is not available on this system" IS part of the answer.
        return None, {"refused": "ValueError: {}".format(exc),
                      "wall_s": round(time.time() - t0, 3),
                      "cpu_s": round(time.process_time() - c0, 3)}
    r = solver.last
    return np.asarray(r.energies, dtype=float), {
        "n_sweeps": int(r.n_sweeps), "max_bond_dim": int(r.max_bond_dim),
        "w_disc": float(getattr(r, "discarded_weight", float("nan"))),
        "wall_s": round(time.time() - t0, 3), "cpu_s": round(time.process_time() - c0, 3),
    }


def protocol_casci(key: str, *, memory_gb: float) -> List[Dict]:
    """One sweep at fixed guess orbitals against exact CI on the same integrals."""
    n_elec, n_act, n_roots, cap_sat, cap_trunc = [c[1:] for c in CASES
                                                  if c[0] == key][0]
    system = _system(key)
    res.clear()
    reference, space = _reference_and_space(system, n_act, n_elec, memory_gb=memory_gb)
    ints = _cas_integrals(reference, space)
    e_ci, ci_wall, ci_cpu = _exact_ci(ints, n_elec, n_roots)
    ci_spread = _group_spread(e_ci, n_roots)
    print("    exact CI: group spread {:.3e} Eh  ({:.2f} s wall, {:.2f} s cpu)".format(
        ci_spread, ci_wall, ci_cpu), flush=True)
    rows = []
    for cap, label in ((cap_sat, "saturating"), (cap_trunc, "truncating")):
        for conv in CONV_TOLS:
            for dav in DAVIDSON_TOLS:
                e, meta = _sweep(ints, n_elec, n_roots, max_bond=cap, conv_tol=conv,
                                 davidson_tol=dav)
                row = {"key": key, "protocol": "casci", "cap": int(cap), "cap_role": label,
                       "conv_tol": float(conv), "davidson_tol": float(dav),
                       "n_roots": int(n_roots), "ci_group_spread_eh": ci_spread,
                       "ci_wall_s": round(ci_wall, 3), "ci_cpu_s": round(ci_cpu, 3)}
                row.update(meta)
                if e is not None:
                    # ⚠ the sweep's energies EXCLUDE e_core; the CI's total_energies include
                    # it. A spread is blind to the shift, an absolute comparison is not.
                    row["group_spread_eh"] = _group_spread(e, n_roots)
                    row["e_sa_error_eh"] = float((e[:n_roots] + ints.e_core).mean()
                                                 - e_ci[:n_roots].mean())
                    print("    D={:3d} ({:10s}) conv={:.0e} dav={:.0e}  spread={:.3e} Eh  "
                          "dE_SA={:+.2e}  {:2d} sweeps  {:.2f} s cpu".format(
                              cap, label, conv, dav, row["group_spread_eh"],
                              row["e_sa_error_eh"], row["n_sweeps"], row["cpu_s"]),
                          flush=True)
                else:
                    print("    D={:3d} ({:10s}) conv={:.0e} dav={:.0e}  {}".format(
                        cap, label, conv, dav,
                        "REFUSED (degenerate Schmidt group)" if "refused" in meta
                        else "DID NOT CONVERGE"), flush=True)
                rows.append(row)
    return rows


def protocol_caps(key: str, *, memory_gb: float) -> List[Dict]:
    """Spread against the BOND CAP at a fixed convergence criterion.

    The `casci` protocol varies the convergence criterion and holds the cap; this holds the
    criterion and varies the cap. Together they separate the two candidate causes of a
    residual spread — an unconverged sweep, and a truncated one — which is the question.
    """
    n_elec, n_act, n_roots, cap_sat, _ = [c[1:] for c in CASES if c[0] == key][0]
    system = _system(key)
    res.clear()
    reference, space = _reference_and_space(system, n_act, n_elec, memory_gb=memory_gb)
    ints = _cas_integrals(reference, space)
    e_ci, ci_wall, ci_cpu = _exact_ci(ints, n_elec, n_roots)
    ci_spread = _group_spread(e_ci, n_roots)
    print("    exact CI: group spread {:.3e} Eh".format(ci_spread), flush=True)
    rows = []
    for cap in (2, 3, 4, 6, 8, 12, 16, 24, 32):
        if cap > cap_sat * 2:
            continue
        e, meta = _sweep(ints, n_elec, n_roots, max_bond=cap, conv_tol=1e-9,
                         davidson_tol=1e-8)
        row = {"key": key, "protocol": "caps", "cap": int(cap), "conv_tol": 1e-9,
               "n_roots": int(n_roots), "ci_group_spread_eh": ci_spread}
        row.update(meta)
        if e is not None:
            row["group_spread_eh"] = _group_spread(e, n_roots)
            row["e_sa_error_eh"] = float((e[:n_roots] + ints.e_core).mean()
                                         - e_ci[:n_roots].mean())
            print("    D={:3d}  spread={:.3e} Eh  dE_SA={:+.3e} Eh  D_used={:3d}  "
                  "{:2d} sweeps  {:.2f} s cpu".format(
                      cap, row["group_spread_eh"], row["e_sa_error_eh"],
                      row["max_bond_dim"], row["n_sweeps"], row["cpu_s"]), flush=True)
        else:
            print("    D={:3d}  {}".format(cap, "REFUSED (degenerate Schmidt group)"
                                           if "refused" in meta else "DID NOT CONVERGE"),
                  flush=True)
        rows.append(row)
    return rows


def protocol_casscf(key: str, *, memory_gb: float) -> List[Dict]:
    """The validation record's own protocol: a full state-averaged optimization."""
    import kuiva
    from kuiva import CASSCF, Reference, ScalarSCF

    n_elec, n_act, n_roots, cap_sat, cap_trunc = [c[1:] for c in CASES
                                                  if c[0] == key][0]
    system = _system(key)
    res.clear()
    molecule = kuiva.Molecule(atoms=system.atoms, basis=system.basis,
                              charge=system.charge, spin=system.spin)
    scf = ScalarSCF(molecule, memory_gb=memory_gb, screening="none").run()
    ref = Reference(scf).run()
    kw = dict(character=(system.atoms[0][0], system.active_l)) if system.active_l else {}
    rows = []
    # the CI path first: the bar, on the same orbitals-and-protocol
    t0, c0 = time.time(), time.process_time()
    cas = CASSCF(ref, n_active=n_act, n_active_elec=n_elec, n_states=n_roots,
                 report=False, **kw).run()
    e_ci = np.asarray(cas.energies, dtype=float)
    ci_spread = _group_spread(e_ci, n_roots)
    rows.append({"key": key, "protocol": "casscf", "solver": "ci",
                 "group_spread_eh": ci_spread, "n_roots": int(n_roots),
                 "wall_s": round(time.time() - t0, 2),
                 "cpu_s": round(time.process_time() - c0, 2)})
    print("    CI   spread={:.3e} Eh  ({:.1f} s cpu)".format(ci_spread, rows[-1]["cpu_s"]),
          flush=True)
    for conv in CONV_TOLS:
        t0, c0 = time.time(), time.process_time()
        try:
            cas = CASSCF(ref, n_active=n_act, n_active_elec=n_elec, n_states=n_roots,
                         solver="dmrg",
                         solver_options=dict(max_bond=cap_sat, conv_tol=conv,
                                             davidson_tol=min(1e-8, conv * 10)),
                         report=False, **kw).run()
            e = np.asarray(cas.energies, dtype=float)
            row = {"key": key, "protocol": "casscf", "solver": "dmrg", "cap": int(cap_sat),
                   "conv_tol": float(conv), "n_roots": int(n_roots),
                   "group_spread_eh": _group_spread(e, n_roots),
                   "e_error_vs_ci_eh": float(e[:n_roots].mean() - e_ci[:n_roots].mean()),
                   "wall_s": round(time.time() - t0, 2),
                   "cpu_s": round(time.process_time() - c0, 2)}
            print("    dmrg conv={:.0e} spread={:.3e} Eh  dE_SA={:+.2e}  ({:.1f} s cpu)"
                  .format(conv, row["group_spread_eh"], row["e_error_vs_ci_eh"],
                          row["cpu_s"]), flush=True)
        except Exception as exc:                                  # noqa: BLE001
            row = {"key": key, "protocol": "casscf", "solver": "dmrg", "cap": int(cap_sat),
                   "conv_tol": float(conv), "failed": "{}: {}".format(type(exc).__name__,
                                                                      exc),
                   "wall_s": round(time.time() - t0, 2),
                   "cpu_s": round(time.process_time() - c0, 2)}
            print("    dmrg conv={:.0e} FAILED: {}".format(conv, exc), flush=True)
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--protocol", required=True,
                    choices=("casci", "casscf", "caps"))
    ap.add_argument("--out", default="temp/dmrg_degeneracy.json")
    ap.add_argument("--only", default="", help="comma-separated system keys")
    ap.add_argument("--memory-gb", type=float, default=6.0)
    args = ap.parse_args()
    only = {s for s in args.only.split(",") if s}

    data: Dict[str, object] = {"conv_tols": list(CONV_TOLS),
                               "davidson_tols": list(DAVIDSON_TOLS), "records": {}}
    if os.path.exists(args.out):
        with open(args.out) as fh:
            data = json.load(fh)
    data.setdefault("records", {})
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    hb = Heartbeat("dmrg_degeneracy_" + args.protocol, budget_seconds=WALL_BUDGET_S,
                   meta={"protocol": args.protocol})
    for i, case in enumerate(CASES):
        key = case[0]
        if only and key not in only:
            continue
        if hb.expired:
            hb.tick(i, stage="wall budget exhausted before {}".format(key))
            print("wall budget exhausted before {}".format(key))
            break
        hb.tick(i, stage="{}: {}".format(args.protocol, key))
        print("  {} [{}]".format(key, args.protocol), flush=True)
        runner = {"casci": protocol_casci, "caps": protocol_caps,
                  "casscf": protocol_casscf}[args.protocol]
        rows = runner(key, memory_gb=args.memory_gb)
        data["records"].setdefault(args.protocol, {})[key] = rows
        with open(args.out + ".tmp", "w") as fh:
            json.dump(data, fh, indent=1, sort_keys=True)
        os.replace(args.out + ".tmp", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
