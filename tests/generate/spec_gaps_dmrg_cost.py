"""How long one DMRG solve takes, as a function of size.

⚠ **A study generator, not a reference generator.** The question it exists to answer is the
one that decided whether the network state gets its own checkpoint format:
*is a single DMRG solve, or a DMRG-CASSCF built out of them, long enough that losing it
costs more than the format costs?* The criterion was declared before the run — one solve over
~30 minutes, or a whole optimization over a few hours — and the answer is written up in
the local validation notes.

The existing record already covers the small end (recorded: 3-sweep solves of 4–90 s at n = 10–12, D ≤ 48), and the notes record that the n = 30 ab
initio TTNO does not fit a 15 GB machine at all. What is missing is the middle — the cost per
sweep as `n` and `D` grow toward a production active space — and that is all this measures.

Method: a **fixed** sweep count so termination never depends on
convergence, a hard wall budget that kills the ladder rather than the run, incremental
writes, and cost read as CPU seconds. Ambient BLAS is pinned to one thread and the
compiled kernels get the budget — the measured thread policy for this layer, so the numbers are
what a correctly-run production sweep would cost rather than four times it.

Usage::

    python tests/generate/spec_gaps_dmrg_cost.py [--budget 540] [--sweeps 2]

Record: ``temp/spec_gaps_dmrg_cost.json``.
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
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
if "numpy" in sys.modules:                                              # pragma: no cover
    raise SystemExit("numpy was imported before the thread pins could be set; run this "
                     "script directly")

import numpy as np                                                      # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT = REPO / "temp/spec_gaps_dmrg_cost.json"

#: (label, n_spinor, n_elec, modes per node, max_bond, n_roots). Ordered cheapest first so a
#: budget kill leaves the ladder's low rungs measured rather than nothing.
LADDER = (
    ("n=12, 1/node, k=6,  D=32",  12, 6, 1, 32, 2),
    ("n=12, 3/node, k=6,  D=32",  12, 6, 3, 32, 2),
    ("n=16, 1/node, k=8,  D=32",  16, 8, 1, 32, 2),
    ("n=16, 2/node, k=8,  D=64",  16, 8, 2, 64, 2),
    ("n=20, 1/node, k=10, D=64",  20, 10, 1, 64, 2),
    ("n=20, 2/node, k=10, D=128", 20, 10, 2, 128, 2),
    ("n=24, 2/node, k=12, D=128", 24, 12, 2, 128, 2),
)


def random_spinor_integrals(n, seed=0):
    """4-fold-symmetric complex integrals, as the dmrg test suite builds them."""
    rng = np.random.default_rng(seed)
    h = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    h = 0.5 * (h + h.conj().T)
    eri = (rng.standard_normal((n, n, n, n)) + 1j * rng.standard_normal((n, n, n, n)))
    eri = 0.5 * (eri + eri.transpose(2, 3, 0, 1))
    eri = 0.5 * (eri + eri.conj().transpose(1, 0, 3, 2))
    return h, np.ascontiguousarray(eri)


def run_rung(label, n, k, modes_per_node, max_bond, n_roots, sweeps, threads_n):
    from kuiva.dmrg.graph import NetworkGraph
    from kuiva.dmrg.sweep import random_state, solve_ttn, state_gb
    from kuiva.dmrg.ttno import compile_ttno, hamiltonian_product_terms
    from kuiva.util import threads

    h, eri = random_spinor_integrals(n, seed=5)
    contents = [list(range(i, i + modes_per_node)) for i in range(0, n, modes_per_node)]
    graph = NetworkGraph.path(len(contents), contents)

    os.environ["KUIVA_NUM_THREADS"] = str(threads_n)
    threads._reset_cache()
    try:
        t0, c0 = time.time(), time.process_time()
        op = compile_ttno(graph, hamiltonian_product_terms(h, eri))
        compile_wall, compile_cpu = time.time() - t0, time.process_time() - c0
        state = random_state(op, k, max_bond, n_roots=n_roots,
                             rng=np.random.default_rng(3))
        t0, c0 = time.time(), time.process_time()
        # on_split="warn": random integrals are not time-reversal symmetric, so the state average's
        # structural refusal does not apply to them (the dmrg suite's own setting).
        result = solve_ttn(op, state, max_bond=max_bond, max_sweeps=sweeps,
                           conv_tol=0.0, boundary_check=0, report=False, on_split="warn")
        wall, cpu = time.time() - t0, time.process_time() - c0
    finally:
        os.environ.pop("KUIVA_NUM_THREADS", None)
        threads._reset_cache()

    return {"shape": label, "n_spinor": n, "n_elec": k, "modes_per_node": modes_per_node,
            "max_bond": max_bond, "n_roots": n_roots, "sweeps": sweeps,
            "ttno_compile_wall": round(compile_wall, 2),
            "ttno_compile_cpu": round(compile_cpu, 2),
            "sweep_wall": round(wall, 2), "sweep_cpu": round(cpu, 2),
            "wall_per_sweep": round(wall / sweeps, 2),
            "cpu_per_wall": round(cpu / wall, 2) if wall else 0.0,
            "state_gb": round(state_gb(result.state), 4),
            "energy_sa": float(np.mean(result.energies))}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--budget", type=float, default=540.0)
    ap.add_argument("--sweeps", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)

    os.environ.setdefault("KUIVA_MEMORY_GB", "8.0")
    import kuiva
    from kuiva.util import native
    native.activate()

    record: Dict = {"schema": 1, "generator": "tests/generate/spec_gaps_dmrg_cost.py",
                    "kuiva_version": kuiva.__version__,
                    "native_available": native.available(),
                    "native_build": native.build_id() if native.available() else None,
                    "ambient_blas_threads": 1, "kernel_threads": args.threads,
                    "sweeps": args.sweeps, "records": []}
    rows: List[Dict] = record["records"]
    t_start = time.time()
    for label, n, k, mpn, d, nr in LADDER:
        left = args.budget - (time.time() - t_start)
        if left <= 0:
            # ⚠ Write before breaking, or the record says nothing about why it is short.
            record["stopped_early"] = "budget exhausted before {!r}".format(label)
            with open(args.out, "w") as fh:
                json.dump(record, fh, indent=1, sort_keys=True, default=float)
            print("budget exhausted before {!r}".format(label), flush=True)
            break
        try:
            row = run_rung(label, n, k, mpn, d, nr, args.sweeps, args.threads)
        except Exception as exc: # a refusal is data, not a crash
            row = {"shape": label, "error": "{}: {}".format(type(exc).__name__, exc)}
        row["elapsed_s"] = round(time.time() - t_start, 1)
        rows.append(row)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(record, fh, indent=1, sort_keys=True, default=float)
        print("[{:6.1f}s] {:28s} {}".format(
            row["elapsed_s"], label,
            row.get("error", "sweep {:.2f} s wall ({:.2f} cpu/wall), TTNO {:.1f} s, "
                             "state {:.3f} GB".format(row.get("wall_per_sweep", 0.0),
                                                      row.get("cpu_per_wall", 0.0),
                                                      row.get("ttno_compile_wall", 0.0),
                                                      row.get("state_gb", 0.0)))),
              flush=True)
    print("\ntotal {:.1f} s".format(time.time() - t_start))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
