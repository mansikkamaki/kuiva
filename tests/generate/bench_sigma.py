"""Profile the sigma vector and close (or open) the C++ port gate.

The gate, stated in advance so it cannot be rationalized afterwards
-------------------------------------------------------------------
    Port the gather/scatter to C++ if it exceeds **25%** of sigma-vector CPU time at the
    largest reachable determinant space, **or** if ``cpu/wall`` on the sigma kernel sits near
    1 on a threaded run (which would mean it is running serial and the GEMM is not hiding it).

Two things this script exists to get right, both about CPU-vs-wall time:

⚠ **CPU seconds, not wall.** This development machine thermally throttles, so a wall-ranked
profile is partly a temperature measurement. Everything below is ranked on ``cpu``, and the
thermal accounting of :mod:`thermal` is printed alongside so a clamped run is visible rather
than mistaken for a slow kernel.

⚠ **``KMP_BLOCKTIME=0`` is not optional here.** ``process_time`` counts spin-waiting OpenMP
threads, so without it a *serial* region that merely follows a threaded BLAS call is charged
for four idling threads and reads ``cpu/wall = 4``. The two gathers are serial NumPy; with
spin-waiting on they look threaded, which inverts the very comparison the gate turns on. This
script sets it in-process before NumPy is imported and refuses to run if it was imported first.

What is measured
----------------
The three regions of the sigma algorithm, separately — ``sigma step 1: gather F``,
``sigma step 2: GEMM``, ``sigma step 3: gather sigma`` — because they are not alike: step 1
reads the CI vector (3 MB at n = 20, L3-resident) while step 3 reads ``G`` (1.1 GB, fully
randomly). A single "gather" number would hide that, and it is the number the gate turns on.

⚠ **Realistic integrals, not random ones.** Finding 2 of the plan's Stage 2/3 notes: a random
Hermitian matrix is not a hard instance of this problem, it is a different one. Timing is less
sensitive to that than convergence is, but the diagonal built here still carries a Fock-like
spread so the sizes and the access pattern are the ones a real calculation has.

Usage (bounded, incremental, self-reporting)::

    python tests/generate/bench_sigma.py                     # the default ladder, ~5 min
    python tests/generate/bench_sigma.py --spaces 20:10 --repeat 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# ⚠ Before NumPy/MKL start a thread pool. See the module docstring: with spin-waiting on, a
# serial gather that follows a threaded GEMM is charged for the idling threads.
os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
if "numpy" in sys.modules:                                              # pragma: no cover
    raise SystemExit("numpy was imported before KMP_BLOCKTIME could be set; run this script "
                     "directly, not through an importer that pulls in numpy first")

import numpy as np                                                      # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import thermal                                                          # noqa: E402

OUT = REPO / "temp/bench_sigma.json"

#: The port gate's threshold, as a number.
GATHER_FRACTION_GATE = 0.25

#: Default ladder. The top rung is set by **residency, not arithmetic**: ``F`` + ``G`` are
#: 2.2 GB at (20, 10) and 10.2 GB at (22, 11), so 20 spinors is the largest space that fits
#: under the committed 8 GB budget of ``defaults.conf``. That is the conventional-CI ceiling
#: this machine can measure, and it is the space the gate is evaluated at.
DEFAULT_SPACES = ((14, 7), (16, 8), (18, 9), (20, 10))


def parse_spaces(text: str):
    if not text:
        return DEFAULT_SPACES
    out = []
    for item in text.split(","):
        n, k = item.split(":")
        out.append((int(n), int(k)))
    return tuple(out)


def realistic_integrals(n: int, seed: int = 0):
    """A Hermitian one-electron matrix with a Fock-like spread, and 4-fold-symmetric ERIs.

    ⚠ Not a physical Hamiltonian, and it does not have to be — this measures *time*, not
    convergence. What it does have to be is the right size, the right dtype and the right
    permutational symmetry, because :class:`kuiva.ci.sigma.SigmaOperator` asserts the last of
    those on construction and a matrix that failed the assertion would never be timed at all.
    """
    rng = np.random.default_rng(seed)
    h = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    h = 0.5 * (h + h.conj().T)
    h += np.diag(np.arange(n, dtype=float) - n / 2.0)          # a Fock-like spread
    # (pq|rs) = (rs|pq) = (qp|sr)* : build from a real Cholesky-like factor, which gives all
    # three at once and is what a real factorized ERI looks like.
    naux = max(2 * n, 8)
    b = rng.standard_normal((naux, n, n)) + 1j * rng.standard_normal((naux, n, n))
    b = 0.5 * (b + b.conj().transpose(0, 2, 1))
    eri = np.einsum("Ppq,Prs->pqrs", b, b.conj()) / naux
    eri = 0.5 * (eri + eri.transpose(2, 3, 0, 1))
    return np.ascontiguousarray(h), np.ascontiguousarray(eri)


def profile_space(n: int, k: int, repeat: int, memory_gb: float) -> Dict:
    """One (n, k) rung: build the space, apply H ``repeat`` times, return the CPU split."""
    from kuiva.ci.sigma import SigmaOperator, sigma_workspace_gb
    from kuiva.ci.strings import CASSpace, cas_dimension
    from kuiva.util import resources as res
    from kuiva.util import timing

    rec: Dict = {"n_spinor": n, "n_elec": k, "ndet": int(cas_dimension(n, k)),
                 "workspace_gb": float(sigma_workspace_gb(n, k)), "repeat": int(repeat)}
    if rec["workspace_gb"] > memory_gb:
        rec["status"] = "skipped: F+G workspace {:.2f} GB exceeds the {:.1f} GB budget " \
                        "(residency, not flops, sets the ceiling)".format(
                            rec["workspace_gb"], memory_gb)
        return rec

    res.BUDGET.clear()
    timing.reset()
    h, eri = realistic_integrals(n)
    space = CASSpace(n, k)
    operator = SigmaOperator(space, h, eri)
    rng = np.random.default_rng(1)
    c = (rng.standard_normal(space.ndet) + 1j * rng.standard_normal(space.ndet))
    c = np.ascontiguousarray(c / np.linalg.norm(c))

    operator(c)                                             # warm the pages, then measure
    timing.reset()
    t0 = time.time()
    for _ in range(repeat):
        operator(c)
    rec["wall_total"] = round(time.time() - t0, 4)

    regions = {}
    for node in timing.REGISTRY.nodes():
        if node.label.startswith("sigma "):
            regions[node.label] = {"calls": node.calls,
                                   "cpu": round(node.cpu, 4),
                                   "wall": round(node.wall, 4),
                                   "cpu_per_wall": round(node.parallel_ratio, 3)}
    rec["regions"] = regions
    total_cpu = sum(r["cpu"] for r in regions.values())
    gather_cpu = sum(r["cpu"] for name, r in regions.items() if "gather" in name)
    rec["cpu_total"] = round(total_cpu, 4)
    rec["gather_cpu_fraction"] = round(gather_cpu / total_cpu, 4) if total_cpu else 0.0
    gemm = regions.get("sigma step 2: GEMM", {})
    rec["gemm_cpu_fraction"] = round(gemm.get("cpu", 0.0) / total_cpu, 4) if total_cpu else 0.0
    rec["gemm_cpu_per_wall"] = gemm.get("cpu_per_wall", 0.0)
    rec["gate_open"] = bool(rec["gather_cpu_fraction"] > GATHER_FRACTION_GATE)
    rec["status"] = "ok"
    del operator, space, h, eri, c
    return rec


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spaces", default="", help="comma-separated n:k, e.g. 18:9,20:10")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--memory-gb", type=float, default=8.0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    os.environ.setdefault("KUIVA_MEMORY_GB", str(args.memory_gb))
    import kuiva

    out: Dict = {"schema": 1, "generator": "tests/generate/bench_sigma.py",
                 "kuiva_version": kuiva.__version__,
                 "gate_fraction": GATHER_FRACTION_GATE,
                 "kmp_blocktime": os.environ.get("KMP_BLOCKTIME"),
                 "environment": thermal.describe_environment(),
                 "records": []}
    records: List[Dict] = []
    for n, k in parse_spaces(args.spaces):
        with thermal.track_resources() as tr:
            rec = profile_space(n, k, args.repeat, args.memory_gb)
        rec["resources"] = tr.as_dict()
        records.append(rec)
        out["records"] = records
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True))
        if rec["status"] != "ok":
            print("[sigma] n={:2d} k={:2d}  {}".format(n, k, rec["status"]), flush=True)
            continue
        print("[sigma] n={:2d} k={:2d} ndet={:>9d} F+G={:6.2f} GB | gathers {:5.1%} of CPU "
              "| GEMM {:5.1%} (cpu/wall {:.2f}) | gate {} ({})".format(
                  n, k, rec["ndet"], rec["workspace_gb"], rec["gather_cpu_fraction"],
                  rec["gemm_cpu_fraction"], rec["gemm_cpu_per_wall"],
                  "OPEN" if rec["gate_open"] else "closed", tr.summary()), flush=True)
        for name, r in sorted(rec["regions"].items()):
            print("           {:28s} cpu {:8.3f} s  wall {:8.3f} s  cpu/wall {:.2f}".format(
                name, r["cpu"], r["wall"], r["cpu_per_wall"]), flush=True)
        if tr.throttled:
            print("  [warn] CPU thermally clamped for {:.0f}% of this rung; judge it on cpu, "
                  "never on wall ".format(100 * (tr.throttle_fraction or 0)),
                  flush=True)

    print("\nwrote {}".format(args.out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
