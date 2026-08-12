"""Which optimizer ``mode`` is right for a full-CI CASSCF (the open optimizer-mode question).

The question, stated as the design leaves it
---------------------------------------
``mode="auto"`` escalates from a quasi-Newton step to the exact second-order step on the
gradient trajectory, and its work metric assumes **a macro-iteration is cheap relative to a
Hessian-vector product** — "true for the cheap CI, false for DMRG". A full CI can sit on either
side of that: at a few hundred determinants a solve is nothing and the assumption holds, while
at 10^4-10^5 determinants a solve is tens of sigma vectors against an HVP's two J/K builds and
it does not.

⚠ **That is a measurement, not an argument**, and the project is explicit that an unprofiled
optimization is a guess. This script makes it, on both sides of the boundary:

* **orbital-dominated** — a heavy element with many rotation parameters and a tiny CI
  (``tlh``: 188 spinors, 9440 complex parameters, 28 determinants);
* **CI-dominated** — Ti(2+) ``CAS(10, 18)``, 43 758 determinants over 84 spinors, the same
  system example 9 uses, where one CI solve costs far more than a Hessian-vector product.

Both are compared in **work units** (macro-iterations plus a weight per HVP) *and* in CPU
seconds (wall time on this machine is partly a thermal measurement), because
the two can disagree and the work metric is the thing under test.

Usage (bounded, incremental)::

    python tests/generate/bench_casscf_mode.py --cases tlh --max-iter 40
    python tests/generate/bench_casscf_mode.py                       # both cases
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

os.environ.setdefault("KMP_BLOCKTIME", "0")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import thermal                                                          # noqa: E402

OUT = REPO / "temp/bench_casscf_mode.json"

MODES = ("auto", "second-order", "quasi-newton")

#: The two cases, chosen to straddle the mode default's assumption. ``ti2p`` is example 9's system with the
#: **full** CI in place of its selected one, which is what makes it CI-dominated.
CASES: Dict[str, Dict] = {
    "tlh": {"note": "orbital dominated: 188 spinors, 9440 parameters, 28 determinants",
            "system": "tlh"},
    "ti2p": {"note": "CI dominated: Ti(2+) CAS(10,18), 43758 determinants over 84 spinors",
             "atoms": [("Ti", (0.0, 0.0, 0.0))], "basis": "x2c-SVPall-2c",
             "charge": 2, "spin": 2, "n_active": 18, "n_active_elec": 10, "n_states": 3},
}


def build_case(name: str, memory_gb: float):
    """``(reference, selection, n_states, label)`` for one case."""
    from kuiva.interface import api
    spec = CASES[name]
    if "system" in spec:
        import systems as sysdef
        s = sysdef.SYSTEMS_BY_KEY[spec["system"]]
        molecule = api.Molecule(atoms=s.atoms, basis=s.basis, charge=s.charge, spin=s.spin)
        reference = api.spinor_reference(molecule, memory_gb=memory_gb, screening="none")
        n_inactive = reference.data.nelec_total - s.nelecas
        window = list(range(n_inactive, n_inactive + 2 * s.ncas))
        return reference, dict(active=window, n_active_elec=s.nelecas), s.soc_states, s.label
    molecule = api.Molecule(atoms=spec["atoms"], basis=spec["basis"],
                            charge=spec["charge"], spin=spec["spin"])
    reference = api.spinor_reference(molecule, memory_gb=memory_gb, screening="none")
    n_inactive = reference.data.nelec_total - spec["n_active_elec"]
    window = list(range(n_inactive, n_inactive + spec["n_active"]))
    return (reference, dict(active=window, n_active_elec=spec["n_active_elec"]),
            spec["n_states"], name)


def run_mode(reference, selection, n_states: int, mode: str, max_iter: int,
             conv_grad: float) -> Dict:
    from kuiva.interface import api
    t_wall, t_cpu = time.time(), time.process_time()
    outcome = api.casscf(reference, n_states=n_states, mode=mode, max_iter=max_iter,
                         conv_grad=conv_grad, report=False, **selection)
    orb = outcome.orbital
    return {"mode": mode, "converged": bool(outcome.converged),
            "iterations": int(orb.n_iterations), "grad_norm": float(orb.grad_norm),
            "energy": float(outcome.energy),
            "hessian_matvec": int(orb.n_hessian_matvec),
            "second_order_steps": int(orb.n_second_order_steps),
            "rejected": int(orb.n_rejected),
            "work_units": float(orb.work_units),
            "wall": round(time.time() - t_wall, 3),
            "cpu": round(time.process_time() - t_cpu, 3),
            "ci_solves": int(getattr(outcome.solver, "n_solves", 0)),
            "sigma_applications": int(outcome.solver.n_apply)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--max-iter", type=int, default=40)
    ap.add_argument("--conv-grad", type=float, default=1e-4)
    ap.add_argument("--memory-gb", type=float, default=8.0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    os.environ.setdefault("KUIVA_MEMORY_GB", str(args.memory_gb))

    import kuiva
    out: Dict = {"schema": 1, "generator": "tests/generate/bench_casscf_mode.py",
                 "kuiva_version": kuiva.__version__, "max_iter": args.max_iter,
                 "conv_grad": args.conv_grad,
                 "environment": thermal.describe_environment(), "records": {}}
    if Path(args.out).is_file():
        try:
            out = json.loads(Path(args.out).read_text())
            out.setdefault("records", {})
        except ValueError:                                              # pragma: no cover
            pass

    for name in [c for c in args.cases.split(",") if c]:
        reference, selection, n_states, label = build_case(name, args.memory_gb)
        rows: List[Dict] = []
        for mode in [m for m in args.modes.split(",") if m]:
            with thermal.track_resources() as tr:
                try:
                    rec = run_mode(reference, selection, n_states, mode, args.max_iter,
                                   args.conv_grad)
                except Exception as exc:                                # noqa: BLE001
                    rec = {"mode": mode, "error": "{}: {}".format(type(exc).__name__, exc)}
            rec["resources"] = tr.as_dict()
            rows.append(rec)
            out["records"][name] = {"note": CASES[name]["note"], "label": label,
                                    "n_states": n_states, "modes": rows}
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True))
            if "error" in rec:
                print("[mode] {:6s} {:14s} ERROR {}".format(name, mode, rec["error"]),
                      flush=True)
                continue
            print("[mode] {:6s} {:14s} conv={:5s} iter={:3d} |g|={:.2e} hvp={:4d} "
                  "work={:7.1f} cpu={:8.1f}s sigma={:6d} E={:.9f}".format(
                      name, mode, str(rec["converged"]), rec["iterations"],
                      rec["grad_norm"], rec["hessian_matvec"], rec["work_units"],
                      rec["cpu"], rec["sigma_applications"], rec["energy"]), flush=True)
            if tr.throttled:
                print("  [warn] thermally clamped for {:.0f}% of this run; compare cpu, not "
                      "wall ".format(100 * (tr.throttle_fraction or 0)),
                      flush=True)

    print("\nwrote {}".format(args.out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
