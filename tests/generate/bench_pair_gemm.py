"""Benchmark the block-pair GEMM: kernel threads vs BLAS threads.

The measurement behind the compiled port of this kernel, on the same shapes as
the recorded sweep-cost protocol: random 4-fold-symmetric complex integrals, fixed sweep
count, (n = 10, one mode per node, k = 5) and (n = 12, three modes per node, k = 6), D = 32.

The finding this either confirms or refutes: **BLAS threading buys this layer nothing**
(identical wall at 1 and 8 threads, cpu/wall 3.99 of spin-wait), because the blocks are far
below intra-GEMM threading granularity — so the parallelism worth having is *across* the
pair table, which is exactly what the native ``block_pair_gemm`` claims. The claim to test,
stated in advance: **wall time falls with kernel threads for the first time in this layer**,
and the 4-thread wall approaches the serial wall divided by the parallelizable fraction of
the pair table.

Ambient BLAS is pinned to **one** thread for every variant (`MKL_NUM_THREADS=1`), for two
reasons: measurement already established that BLAS width does not move this layer's wall, and a
single-width ambient makes ``cpu/wall`` read as kernel parallelism rather than spin-wait
. Variants: numpy; native with the kernel thread budget at 1 / 2 / 4.

Usage::

    python tests/generate/bench_pair_gemm.py            # both shapes, ~4 CPU min
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
# Pin ambient BLAS to one thread BEFORE NumPy loads (see the module docstring). The kernel's
# own OpenMP team is unaffected: its size is the explicit n_threads argument.
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

import thermal                                                          # noqa: E402

OUT = REPO / "temp/bench_pair_gemm.json"

MAX_BOND = 32
SWEEPS = 3
N_ROOTS = 2

#: (label, n_spinor, k, modes-per-node). The two recorded sweep-cost shapes.
SHAPES = (("n=10, 1 mode/node", 10, 5, 1),
          ("n=12, 3 modes/node", 12, 6, 3))

#: (backend, kernel threads). The numpy kernel ignores its thread argument by contract.
VARIANTS = (("numpy", 1), ("native", 1), ("native", 2), ("native", 4))


def random_spinor_integrals(n, seed=0):
    """4-fold-symmetric complex integrals, as the dmrg test suite builds them."""
    rng = np.random.default_rng(seed)
    h = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    h = 0.5 * (h + h.conj().T)
    eri = (rng.standard_normal((n, n, n, n)) + 1j * rng.standard_normal((n, n, n, n)))
    eri = 0.5 * (eri + eri.transpose(2, 3, 0, 1))
    eri = 0.5 * (eri + eri.conj().transpose(1, 0, 3, 2))
    return h, np.ascontiguousarray(eri)


def run_variant(label: str, n: int, k: int, modes_per_node: int, backend: str,
                n_threads: int) -> Dict:
    from kuiva.ci import kernels
    from kuiva.dmrg.graph import NetworkGraph
    from kuiva.dmrg.sweep import random_state, solve_ttn
    from kuiva.dmrg.ttno import compile_ttno, hamiltonian_product_terms
    from kuiva.util import threads, timing

    h, eri = random_spinor_integrals(n, seed=5)
    contents = [list(range(i, i + modes_per_node))
                for i in range(0, n, modes_per_node)]
    graph = NetworkGraph.path(len(contents), contents)

    previous = kernels.set_preferred_backend(backend)
    os.environ["KUIVA_NUM_THREADS"] = str(n_threads)
    threads._reset_cache()
    timing.reset()
    try:
        op = compile_ttno(graph, hamiltonian_product_terms(h, eri))
        state = random_state(op, k, MAX_BOND, n_roots=N_ROOTS,
                             rng=np.random.default_rng(3))
        t0 = time.time()
        c0 = time.process_time()
        # on_split="warn": random integrals are not time-reversal symmetric, so the state-averaging
        # structural refusal for odd-electron systems does not apply to them — the same
        # setting the dmrg test suite uses for random-integral runs.
        result = solve_ttn(op, state, max_bond=MAX_BOND, max_sweeps=SWEEPS,
                           conv_tol=0.0, boundary_check=0, report=False, on_split="warn")
        wall = time.time() - t0
        cpu = time.process_time() - c0
    finally:
        kernels.set_preferred_backend(previous)
        os.environ.pop("KUIVA_NUM_THREADS", None)
        threads._reset_cache()
    return {"shape": label, "backend": backend, "n_threads": n_threads,
            "sweeps": SWEEPS, "max_bond": MAX_BOND, "n_roots": N_ROOTS,
            "wall": round(wall, 3), "cpu": round(cpu, 3),
            "cpu_per_wall": round(cpu / wall, 2) if wall else 0.0,
            "energy_sa": float(np.mean(result.energies))}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    os.environ.setdefault("KUIVA_MEMORY_GB", "8.0")
    import kuiva
    from kuiva.util import native

    native.activate()
    if not native.available():
        raise SystemExit("the native backend is not built; run "
                         "`bash scripts/bootstrap/95_native.sh` first")

    record: Dict = {"schema": 1, "generator": "tests/generate/bench_pair_gemm.py",
                    "kuiva_version": kuiva.__version__,
                    "native_build": native.build_id(),
                    "ambient_blas_threads": 1,
                    "kmp_blocktime": os.environ.get("KMP_BLOCKTIME"),
                    "environment": thermal.describe_environment(), "records": []}
    records: List[Dict] = []
    reference: Dict[str, float] = {}
    for label, n, k, modes in SHAPES:
        for backend, n_threads in VARIANTS:
            with thermal.track_resources() as tr:
                rec = run_variant(label, n, k, modes, backend, n_threads)
            rec["resources"] = tr.as_dict()
            if label not in reference:
                reference[label] = rec["energy_sa"]
            rec["energy_error"] = abs(rec["energy_sa"] - reference[label])
            records.append(rec)
            record["records"] = records
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(record, indent=2, sort_keys=True))
            print("[gemm] {:20s} {:6s} nt={} | wall {:8.2f} s  cpu {:8.2f} s "
                  "(cpu/wall {:4.2f}) | dE vs numpy {:.2e} | {}".format(
                      label, backend, n_threads, rec["wall"], rec["cpu"],
                      rec["cpu_per_wall"], rec["energy_error"], tr.summary()),
                  flush=True)
            if tr.throttled:
                print(" [warn] thermally clamped for {:.0f}%: judge on "
                      "cpu".format(100 * (tr.throttle_fraction or 0)), flush=True)
    print("\nwrote {}".format(args.out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
