"""Measure what a whole automatic active-space selection costs, probe by probe, and decide.

What this answers
-----------------
``kuiva/autocas/protocol.py`` runs one cheap-CI probe per round (plus one per pruning that
removed something). Two things about it can only be measured:

* **the cost of a round as the candidate space grows** -- which is also what bounds the
  ``solver="dmrg"`` size budget, because every round is a cheap-CI probe whatever the
  production solver is: the selection cannot hand on a space it could not afford to probe;
* **the verdict of a class that is not a foregone conclusion** -- the correlating (double)
  shell of a lanthanide, which only the spectrum test can accept, since at the probe's level
  of correlation it carries no occupation to see.

Every probe is logged as it finishes (the probe function is wrapped, so a run that is killed
still leaves every probe it had paid for), and the run carries a heartbeat.

Usage (one invocation is one system and one target set, each inside the ten-minute rule)::

    source setup.sh
    python tests/generate/measure_autocas_rounds.py cecl3 --targets double --out temp/x.jsonl
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "generate"))

import systems as sysdef                                                    # noqa: E402
from progress import Heartbeat                                              # noqa: E402

from kuiva.autocas import protocol as proto                                 # noqa: E402
from kuiva.autocas.probe import ProbeBudget                                 # noqa: E402
from kuiva.interface import api                                             # noqa: E402

#: Named target sets, so a run line in the notes says exactly what was asked for.
TARGETS = {
    "shells": lambda element: "shells",
    "double": lambda element: ["shells", ("double", element)],
    "bonding": lambda element: ["shells", ("bonding", element, 4)],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("system")
    parser.add_argument("--targets", default="shells", choices=sorted(TARGETS))
    parser.add_argument("--element", default=None,
                        help="the centre's element for the class targets (default: the "
                             "system's first atom)")
    parser.add_argument("--max-iter", type=int, default=8)
    parser.add_argument("--out", required=True)
    parser.add_argument("--wall", type=float, default=560.0)
    args = parser.parse_args()

    system = sysdef.get(args.system)
    element = args.element or "".join(ch for ch in system.atoms[0][0] if ch.isalpha())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    hb = Heartbeat("autocas_rounds_" + args.system, budget_seconds=args.wall,
                   meta=dict(targets=args.targets))

    def record(**fields):
        with out.open("a") as fh:
            fh.write(json.dumps(fields) + "\n")

    t0 = time.time()
    molecule = api.Molecule(atoms=system.atoms, basis=system.basis, charge=system.charge,
                            spin=system.spin)
    data = api.scalar_x2c_reference(molecule, screening="none", memory_gb=8.0,
                                    atomic_reference=True)
    reference = api.spinor_reference(data, memory_gb=8.0)
    record(event="front end", system=args.system, wall=time.time() - t0)

    original = proto.probe
    count = [0]

    def logged(ref, coeff, space, **kwargs):
        # ⚠ The wall budget is checked BEFORE a probe is paid for: a probe cannot be
        # interrupted, so the only honest stop is not to start one.
        if hb.expired:
            raise SystemExit("wall budget spent before probe {}".format(count[0]))
        result = original(ref, coeff, space, **kwargs)
        count[0] += 1
        record(event="probe", index=count[0], n_active=int(space.n_active),
               n_elec=int(space.n_elec), n_roots=int(result.n_roots),
               n_determinants=int(result.n_determinants), cpu=result.cpu_seconds,
               wall=result.wall_seconds, converged=bool(result.converged),
               spectrum_cm=[float(x) for x in result.spectrum_cm[:24]])
        hb.tick(count[0], n_active=int(space.n_active), cpu=result.cpu_seconds)
        return result

    proto.probe = logged
    assembly = proto.assemble(reference, targets=TARGETS[args.targets](element),
                              probe_budget=ProbeBudget(max_iter=args.max_iter), report=True)
    for r in assembly.rounds:
        record(event="round", index=r.index, cls=r.cls, candidates=r.n_candidates,
               pruned=r.n_pruned, d_cm=r.distance_cm, tol_cm=r.tolerance_cm,
               verdict=r.verdict, n_spinors=r.n_spinors, cpu=r.cpu_seconds, note=r.note)
    record(event="done", n_active=assembly.n_active, n_elec=int(assembly.space.n_elec),
           floor=assembly.floor, proposal=assembly.proposal.n_states,
           budget=assembly.budget.describe(), wall=time.time() - t0)
    hb.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
