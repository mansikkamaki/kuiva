"""Where the cheap CI's memory goes at a Tier-3 determinant count, phase by phase.

What this answers
-----------------
A probe of ``mn3_linear``'s CAS(15, 30) at 40 000 determinants reached a 13.5 GB resident
high-water mark and was killed, while the pre-flight had planned nothing for it: the pair list
and the sparse CI Hamiltonian are checked only as they are built, and whatever peaked was not
checked at all. This runs the cheap CI's own phases one at a time on the stored active-space
integrals of that system (no front end) and records the resident high-water mark of each,
resetting it between phases (``/proc/self/clear_refs``), with the phase's CPU and wall time.

Usage::

    source setup.sh
    python tests/generate/measure_cheap_ci_memory.py --max-determinants 12000 \\
        --out temp/cheap_ci_memory/mn3_12000.jsonl
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from kuiva.ci.strings import connections, hamiltonian_matrix, rdm12        # noqa: E402
from kuiva.mcscf import preopt as po                                        # noqa: E402
from kuiva.util.logging import set_verbosity                                # noqa: E402

INTEGRALS = ROOT / "temp" / "dmrg_campaign" / "mn3_linear_tier3_integrals.npz"


def hwm_gb() -> float:
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024.0 ** 2
    return float("nan")


def reset_hwm() -> None:
    try:
        with open("/proc/self/clear_refs", "w") as fh:
            fh.write("5")
    except OSError:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-determinants", type=int, default=6000)
    parser.add_argument("--n-states", type=int, default=24)
    parser.add_argument("--integrals", default=str(INTEGRALS))
    args = parser.parse_args()
    set_verbosity("WARNING")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("")

    def phase(name, fn):
        reset_hwm()
        base = hwm_gb()
        c0, t0 = time.process_time(), time.time()
        value = fn()
        row = dict(phase=name, hwm_gb=round(hwm_gb(), 3), base_gb=round(base, 3),
                   cpu_s=round(time.process_time() - c0, 1), wall_s=round(time.time() - t0, 1))
        with out.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print("#", json.dumps(row), flush=True)
        return value

    z = np.load(args.integrals)
    h, eri, n_elec = z["h"], z["eri"], int(z["n_elec"])
    n_spinor = h.shape[0]
    max_det, n_states = args.max_determinants, args.n_states
    with out.open("a") as fh:
        fh.write(json.dumps(dict(max_determinants=max_det, n_states=n_states,
                                 cas=[n_elec, n_spinor], pid=os.getpid())) + "\n")

    dets = phase("reference CAS", lambda: po.reference_determinants(
        np.diag(h), n_spinor, n_elec, max_reference=min(po.DEFAULT_MAX_REFERENCE, max_det)))
    energies, civecs = phase("reference solve", lambda: po._solve(dets, h, eri, n_states))
    weights = np.full(n_states, 1.0 / n_states)
    for rnd in range(2):
        grown = phase("selection round {}".format(rnd + 1), lambda: po._select_space(
            dets, civecs, energies, h, eri, max_det, 2, po.DEFAULT_MAX_GENERATORS, weights))
        if grown.ndet == dets.ndet:
            break
        dets = grown
        conn = phase("connections ({} dets)".format(dets.ndet), lambda: connections(dets))
        with out.open("a") as fh:
            fh.write(json.dumps(dict(n_single=conn.n_single, n_double=conn.n_double)) + "\n")
        hmat = phase("Hamiltonian", lambda: hamiltonian_matrix(dets, h, eri, conn))
        with out.open("a") as fh:
            fh.write(json.dumps(dict(nnz=int(hmat.nnz))) + "\n")
        del hmat
        energies, civecs = phase("solve ({} dets)".format(dets.ndet),
                                 lambda: po._solve(dets, h, eri, n_states, conn=conn))
    phase("RDMs", lambda: rdm12(dets, civecs, weights, conn))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
