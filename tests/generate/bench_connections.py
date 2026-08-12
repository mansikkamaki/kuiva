"""Benchmark the determinant connection scan: NumPy vs the compiled backend.

The measurement behind the compiled port of this kernel, and the record behind the
measurement that closes it: the **cheap CI on TiCl3** — the system
whose 3432-determinant run is the anchor of the recorded CASSCF benchmarks — at
determinant budgets 2 000 / 8 000 / 20 000, under three kernel configurations:

* ``numpy``            — the reference implementation (bitwise-identical answers);
* ``native``, 1 thread — what a serial translation buys (the Python bookkeeping);
* ``native``, 4 threads — what the threaded scan buys on top (the dev-box thread cap).

What is timed is the ``determinant connections`` region of one full :func:`cheap_ci`
selection — the real consumer (selection growth, generators, the final space, all its
re-scans), on a CAS(11,20) frontier window whose 167 960 candidate determinants make
every budget a genuine truncation. The numbers state directly what determinant budget is
affordable inside the cheap-CI selection loop, which is the practical question the port
answers.

House rules: ``KMP_BLOCKTIME=0`` before NumPy loads; CPU seconds are
the cost and wall is reported beside them; results are written incrementally; the front-end
(SCF + integrals) runs **once** and is reused, with ``screening="none"`` because the SOC
screening changes no scalar quantity here and its four-component atomic solves are pure
cost on a timing run.

Usage::

    python tests/generate/bench_connections.py                 # full ladder, ~5 min
    python tests/generate/bench_connections.py --budgets 2000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# ⚠ Before NumPy/MKL start a thread pool (spin-waiting threads are charged to
# whatever serial region follows a threaded call, and this benchmark is *about* a serial
# region on one of its three rungs).
os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
if "numpy" in sys.modules:                                              # pragma: no cover
    raise SystemExit("numpy was imported before KMP_BLOCKTIME could be set; run this "
                     "script directly")

import numpy as np                                                      # noqa: E402,F401

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import thermal                                                          # noqa: E402
from systems import SYSTEMS_BY_KEY                                      # noqa: E402

OUT = REPO / "temp/bench_connections.json"

DEFAULT_BUDGETS = (2000, 8000, 20000)

#: (backend, n_threads). The numpy row ignores its thread count by contract.
VARIANTS = (("numpy", 1), ("native", 1), ("native", 4))


#: The candidate window: 11 electrons in the 20 frontier spinors (the Ti 3d shell plus the
#: upper ligand valence). C(20,11) = 167 960 candidate determinants, so every budget below
#: genuinely truncates and the O(N^2) selection search is the dominant cost — which is the
#: regime the port is for. 62 inactive spinors keep the Kramers-pair boundary even.
N_ACT_ELEC, N_ACT_SPINOR = 11, 20


def build_reference(memory_gb: float):
    """The TiCl3 front-end and CAS integrals, once: identical under every variant."""
    from kuiva.interface.api import Molecule, spinor_reference
    from kuiva.mcscf.orbopt import CASIntegrals, OrbitalSpaces

    system = SYSTEMS_BY_KEY["ticl3"]
    mol = Molecule(atoms=system.atoms, basis=system.basis, charge=system.charge,
                   spin=system.spin)
    ref = spinor_reference(mol, memory_gb=memory_gb, screening="none")
    n_inactive = ref.data.nelec_total - N_ACT_ELEC
    assert n_inactive % 2 == 0
    spaces = OrbitalSpaces.from_counts(n_inactive, N_ACT_SPINOR, ref.nspinor)
    ints = CASIntegrals.build(ref.factors, ref.h_one_electron(), ref.spinors_in_ao(),
                              spaces, e_nuc=ref.data.e_nuc)
    return ints.h_active_effective(), ints.active_eri()


def run_variant(h_eff, eri, budget: int, backend: str, n_threads: int) -> Dict:
    from kuiva.ci import kernels
    from kuiva.mcscf.preopt import cheap_ci
    from kuiva.util import threads, timing

    previous = kernels.set_preferred_backend(backend)
    os.environ["KUIVA_NUM_THREADS"] = str(n_threads)
    threads._reset_cache()
    timing.reset()
    try:
        t0 = time.time()
        c0 = time.process_time()
        result = cheap_ci(h_eff, eri, N_ACT_ELEC, n_states=2, max_determinants=budget)
        wall_total = time.time() - t0
        cpu_total = time.process_time() - c0
    finally:
        kernels.set_preferred_backend(previous)
        os.environ.pop("KUIVA_NUM_THREADS", None)
        threads._reset_cache()
    rec = {"backend": backend, "n_threads": n_threads, "budget": budget,
           "wall_total": round(wall_total, 3), "cpu_total": round(cpu_total, 3),
           "energy": float(np.dot(result.weights, result.energies)),
           "n_determinants": int(result.n_determinants)}
    # ⚠ the timer nests by call path, so the label appears as SEVERAL nodes (reference
    # space, each selection growth, the final space): sum them, or the number reported is
    # whichever single node came first — a mistake this script shipped with once.
    calls = cpu = wall = 0.0
    for node in timing.REGISTRY.nodes():
        if node.label == "determinant connections":
            calls += node.calls
            cpu += node.cpu
            wall += node.wall
    rec["connections"] = {"calls": int(calls), "cpu": round(cpu, 4),
                          "wall": round(wall, 4),
                          "cpu_per_wall": round(cpu / wall, 3) if wall else 0.0}
    return rec


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budgets", default="",
                    help="comma-separated determinant budgets, default 2000,8000,20000")
    ap.add_argument("--memory-gb", type=float, default=8.0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    budgets = tuple(int(b) for b in args.budgets.split(",")) if args.budgets \
        else DEFAULT_BUDGETS

    os.environ.setdefault("KUIVA_MEMORY_GB", str(args.memory_gb))
    import kuiva
    from kuiva.util import native

    native.activate()
    if not native.available():
        raise SystemExit("the native backend is not built; run "
                         "`bash scripts/bootstrap/95_native.sh` first — this benchmark "
                         "exists to compare it against numpy")

    record: Dict = {"schema": 1, "generator": "tests/generate/bench_connections.py",
                    "kuiva_version": kuiva.__version__,
                    "native_build": native.build_id(),
                    "kmp_blocktime": os.environ.get("KMP_BLOCKTIME"),
                    "environment": thermal.describe_environment(), "records": []}
    h_eff, eri = build_reference(args.memory_gb)
    records: List[Dict] = []
    reference_energy: Dict[int, float] = {}
    for budget in budgets:
        for backend, n_threads in VARIANTS:
            with thermal.track_resources() as tr:
                rec = run_variant(h_eff, eri, budget, backend, n_threads)
            rec["resources"] = tr.as_dict()
            # The parity statement that makes the cost columns comparable: every variant of
            # a budget selects the SAME space (the scan is bitwise, asserted in the suite)
            # and lands on the same energy to solver noise. ⚠ Exact equality is the wrong
            # check here: eigsh's random start vectors make two solves of the *same*
            # Hamiltonian in one process differ at ~1e-13, backend regardless — measured on
            # two consecutive native runs before this comparison was loosened.
            key = budget
            if key not in reference_energy:
                reference_energy[key] = rec["energy"]
            rec["energy_delta"] = abs(rec["energy"] - reference_energy[key])
            rec["energy_matches_numpy"] = bool(rec["energy_delta"] < 1e-9)
            records.append(rec)
            record["records"] = records
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(record, indent=2, sort_keys=True))
            conn = rec.get("connections", {})
            print("[conn] budget={:>6d} {:6s} nt={} | connections cpu {:8.3f} s wall "
                  "{:8.3f} s (cpu/wall {:4.2f}, {} calls) | total wall {:7.2f} s | "
                  "energy {} ({})".format(
                      budget, backend, n_threads, conn.get("cpu", float("nan")),
                      conn.get("wall", float("nan")), conn.get("cpu_per_wall", 0.0),
                      conn.get("calls", 0), rec["wall_total"],
                      "match" if rec["energy_matches_numpy"] else "DIFFERS",
                      tr.summary()), flush=True)
            if tr.throttled:
                print("  [warn] thermally clamped for {:.0f}% of this rung ("
                      "14.3): judge on cpu".format(100 * (tr.throttle_fraction or 0)),
                      flush=True)
    print("\nwrote {}".format(args.out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
