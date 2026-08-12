"""Post-port DMRG sweep re-profile.

The `block_pair_gemm` port moved the pair-table loop into the compiled backend and measured a ~30%
parallelizable fraction on the fat-node sweep (measured), so the
next wall-time win in this layer lives *outside* `block_pair_gemm`. This script measures
where, applying the C++ port gate exactly as declared: a region is a candidate only if
it is serial (`cpu/wall ~ 1` at `KMP_BLOCKTIME=0`), exceeds 25% of the stage's CPU seconds
at the largest ad-hoc-reachable size, and its share does not *fall* with size.

Instrument
----------
The profiled stage is ``solve_ttn`` alone — the TTNO compile and the state preparation are
separate stages with their own record. Two runs per configuration, because a profiler is
not a clock:

* an **unprofiled** run whose wall/CPU totals are the authoritative cost (the same
  measurement `bench_pair_gemm.py` makes), and
* a **cProfile** run, on a freshly compiled operator so no lazily built CSR cache is warm,
  whose per-function `tottime` supplies the *shares*. cProfile's per-call overhead inflates
  Python-heavy regions relative to C-heavy ones, so the printed distortion factor
  (profiled wall / unprofiled wall) bounds how far a share can be trusted; shares are
  quoted from this run, absolute seconds never are.

Attribution walks the profiler's caller graph: a kuiva function owns its own `tottime`,
while builtins and NumPy/SciPy internals (`ascontiguousarray`, ufunc reductions, LAPACK,
`csr_matvecs`, ...) are charged to the kuiva region that called them, splitting their time
across callers in proportion to the recorded per-caller time. Nothing may hide in a
generic bucket — the residual is printed, and a surprise there is a finding by
definition.

Ambient BLAS is pinned to **one** thread and the native kernel budget to **one** thread:
with everything serial, `tottime` is CPU attribution and cpu/wall ~ 1 certifies no region
is secretly threaded. The size axis is the bond dimension (D = 16/32/48 on the
fat-node shape) — the axis a production run actually turns — plus the thin-node shape as
the small-block extreme.

Usage::

    python tests/generate/profile_dmrg_sweep.py                 # full ladder, ~10 min
    python tests/generate/profile_dmrg_sweep.py --config 2      # one rung (incremental)
"""
from __future__ import annotations

import argparse
import cProfile
import json
import os
import pstats
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import thermal                                                          # noqa: E402

OUT = REPO / "temp/profile_dmrg_sweep.json"

SWEEPS = 2
N_ROOTS = 2

#: (label, n_spinor, k, modes-per-node, max_bond). The D ladder on the measured fat-node
#: shape is the size axis for the gate's "share does not fall with size" test; the thin
#: shape anchors the small-block regime where the layer was measured overhead-bound.
CONFIGS = (("n=10 thin, D=32", 10, 5, 1, 32),
           ("n=12 fat, D=16", 12, 6, 3, 16),
           ("n=12 fat, D=32", 12, 6, 3, 32),
           ("n=12 fat, D=48", 12, 6, 3, 48))

#: Region buckets for kuiva-owned functions, keyed by (basename, function-name prefix
#: tuple). Generic names (<genexpr>, <listcomp>, <lambda>) and every non-kuiva frame are
#: attributed through the caller graph instead.
_KUIVA_BUCKETS = {
    "block.py": (
        ("pair-GEMM kernel + pack", ("tensordot", "_matricized_buffer",
                                     "block_pair_gemm")),
        ("fuse/split bookkeeping", ("fuse", "split", "_row_keys", "_allowed_rows",
                                    "_flux", "transpose", "_as_matrix")),
        ("svd/qr (incl. LAPACK)", ("svd", "qr")),
        ("BlockTensor construct/arith", ("__init__", "__new__", "_trusted", "zeros",
                                         "random", "find", "copy", "conj", "norm",
                                         "__add__", "__sub__", "__mul__", "nbytes")),
    ),
    "sparse.py": (
        ("sparse-W CSR dot", ("dot_sparse",)),
        ("sparse-W CSR assembly", ("_pattern", "from_entries", "from_block_tensor",
                                   "close_leading_leg", "with_values", "block_entries",
                                   "entry_positions", "dense_block")),
    ),
    "sweep.py": (
        ("environment cache", ("_build", "get", "refresh", "release_all", "_w_lab")),
        ("local solver pack/apply", ("pack", "unpack", "apply", "diagonal", "_env_diag",
                                     "_w_diag")),
        ("tensordot orchestration", ("dot", "to")),
        ("sweep orchestration", ("_update_bond", "_solve_local", "_commit_split",
                                 "_stack_roots", "_boundary_sweep", "solve_ttn",
                                 "_normalized", "_sweep_weights")),
    ),
    "davidson.py": (("davidson eigensolver", ("",)),),
    "kernels.py": (("dispatch shim", ("",)),),
    "native.py": (("dispatch shim", ("",)),),
    "ttno.py": (("ttno (should be idle in a solve)", ("",)),),
    "rdm.py": (("state-average gate", ("",)),),
}

_GENERIC_NAMES = ("<genexpr>", "<listcomp>", "<dictcomp>", "<setcomp>", "<lambda>",
                  "<module>")


def random_spinor_integrals(n, seed=0):
    """4-fold-symmetric complex integrals, as the dmrg test suite builds them."""
    rng = np.random.default_rng(seed)
    h = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    h = 0.5 * (h + h.conj().T)
    eri = (rng.standard_normal((n, n, n, n)) + 1j * rng.standard_normal((n, n, n, n)))
    eri = 0.5 * (eri + eri.transpose(2, 3, 0, 1))
    eri = 0.5 * (eri + eri.conj().transpose(1, 0, 3, 2))
    return h, np.ascontiguousarray(eri)


def _setup(n: int, k: int, modes_per_node: int, max_bond: int):
    from kuiva.dmrg.graph import NetworkGraph
    from kuiva.dmrg.sweep import random_state
    from kuiva.dmrg.ttno import compile_ttno, hamiltonian_product_terms

    h, eri = random_spinor_integrals(n, seed=5)
    contents = [list(range(i, i + modes_per_node))
                for i in range(0, n, modes_per_node)]
    graph = NetworkGraph.path(len(contents), contents)
    op = compile_ttno(graph, hamiltonian_product_terms(h, eri))
    state = random_state(op, k, max_bond, n_roots=N_ROOTS,
                         rng=np.random.default_rng(3))
    return op, state


def _solve(op, state, max_bond: int):
    from kuiva.dmrg.sweep import solve_ttn

    # on_split="warn": random integrals are not time-reversal symmetric (same setting as
    # the dmrg test suite and bench_pair_gemm.py).
    return solve_ttn(op, state, max_bond=max_bond, max_sweeps=SWEEPS, conv_tol=0.0,
                     boundary_check=0, report=False, on_split="warn")


def _own_bucket(func: Tuple[str, int, str]) -> Optional[str]:
    """The bucket a frame owns outright, or ``None`` for attribute-through-callers."""
    filename, _line, name = func
    if "kuiva._native" in name:                 # the compiled kernels are builtin frames
        return ("pair-GEMM kernel + pack" if "block_pair_gemm" in name
                else "native kernel (other)")
    if "kuiva" not in filename:
        return None
    if name in _GENERIC_NAMES:
        return None
    table = _KUIVA_BUCKETS.get(Path(filename).name)
    if table is None:
        return "kuiva other ({})".format(Path(filename).name)
    for bucket, names in table:
        for pat in names:
            if name.startswith(pat):
                return bucket
    return "kuiva other ({})".format(Path(filename).name)


def _resolve_shares(stats: pstats.Stats):
    """Charge every frame's ``tottime`` to a region bucket via the caller graph."""
    table = stats.stats
    memo: Dict[tuple, Dict[str, float]] = {}

    def distribution(func, seen) -> Dict[str, float]:
        if func in memo:
            return memo[func]
        own = _own_bucket(func)
        if own is not None:
            memo[func] = {own: 1.0}
            return memo[func]
        callers = table[func][4]
        live = {c: v for c, v in callers.items() if c not in seen}
        if not live:
            return {"other (residual)": 1.0}
        weights = {c: v[2] for c, v in live.items()}          # per-caller tottime
        total = sum(weights.values())
        if total <= 0.0:
            weights = {c: float(v[0]) for c, v in live.items()}   # fall back: call counts
            total = sum(weights.values()) or 1.0
        dist: Dict[str, float] = {}
        for caller, w in weights.items():
            for bucket, frac in distribution(caller, seen | {func}).items():
                dist[bucket] = dist.get(bucket, 0.0) + (w / total) * frac
        memo[func] = dist
        return dist

    buckets: Dict[str, float] = {}
    rows: List[Tuple[float, str, str]] = []
    total_tt = 0.0
    for func, row in table.items():
        tottime = row[2]
        total_tt += tottime
        dist = distribution(func, frozenset())
        for bucket, frac in dist.items():
            buckets[bucket] = buckets.get(bucket, 0.0) + tottime * frac
        main = max(dist.items(), key=lambda kv: kv[1])[0] if dist else "?"
        rows.append((tottime, "{}:{}".format(Path(func[0]).name, func[2]), main))
    rows.sort(reverse=True)
    return buckets, rows, total_tt


def run_config(label: str, n: int, k: int, modes: int, max_bond: int) -> Dict:
    from kuiva.ci import kernels
    from kuiva.util import threads

    previous = kernels.set_preferred_backend("native")
    os.environ["KUIVA_NUM_THREADS"] = "1"
    threads._reset_cache()
    try:
        # Authoritative cost: unprofiled, cold CSR caches (fresh compile).
        op, state = _setup(n, k, modes, max_bond)
        t0, c0 = time.time(), time.process_time()
        result = _solve(op, state, max_bond)
        wall = time.time() - t0
        cpu = time.process_time() - c0

        # Attribution: profiled, again on a cold operator so lazy CSR builds are seen.
        op, state = _setup(n, k, modes, max_bond)
        prof = cProfile.Profile()
        t1 = time.time()
        prof.enable()
        _solve(op, state, max_bond)
        prof.disable()
        wall_prof = time.time() - t1
    finally:
        kernels.set_preferred_backend(previous)
        os.environ.pop("KUIVA_NUM_THREADS", None)
        threads._reset_cache()

    buckets, rows, total_tt = _resolve_shares(pstats.Stats(prof))
    return {"config": label, "n": n, "k": k, "modes_per_node": modes,
            "max_bond": max_bond, "sweeps": SWEEPS, "n_roots": N_ROOTS,
            "wall": round(wall, 3), "cpu": round(cpu, 3),
            "cpu_per_wall": round(cpu / wall, 2) if wall else 0.0,
            "energy_sa": float(np.mean(result.energies)),
            "profile_distortion": round(wall_prof / wall, 2) if wall else 0.0,
            "profile_total_tottime": round(total_tt, 3),
            "bucket_shares": {name: round(t / total_tt, 4)
                              for name, t in sorted(buckets.items(),
                                                    key=lambda kv: -kv[1])},
            "top_functions": [{"share": round(t / total_tt, 4), "where": w, "bucket": b}
                              for t, w, b in rows[:25] if t / total_tt > 0.002]}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--config", type=int, default=None,
                    help="run a single CONFIGS index (incremental use)")
    args = ap.parse_args(argv)

    os.environ.setdefault("KUIVA_MEMORY_GB", "8.0")
    import kuiva
    from kuiva.util import native

    native.activate()
    if not native.available():
        raise SystemExit("the native backend is not built; run "
                         "`bash scripts/bootstrap/95_native.sh` first")

    out_path = Path(args.out)
    if out_path.exists():
        record = json.loads(out_path.read_text())
    else:
        record = {"schema": 1, "generator": "tests/generate/profile_dmrg_sweep.py",
                  "kuiva_version": kuiva.__version__,
                  "native_build": native.build_id(),
                  "ambient_blas_threads": 1, "kernel_threads": 1,
                  "kmp_blocktime": os.environ.get("KMP_BLOCKTIME"),
                  "environment": thermal.describe_environment(), "records": []}

    todo = CONFIGS if args.config is None else (CONFIGS[args.config],)
    for label, n, k, modes, max_bond in todo:
        with thermal.track_resources() as tr:
            rec = run_config(label, n, k, modes, max_bond)
        rec["resources"] = tr.as_dict()
        record["records"] = [r for r in record["records"] if r["config"] != label]
        record["records"].append(rec)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(record, indent=2, sort_keys=True))
        print("[profile] {:16s} | wall {:7.2f} s  cpu {:7.2f} s (cpu/wall {:4.2f}) "
              "| distortion {:.2f}x | {}".format(
                  label, rec["wall"], rec["cpu"], rec["cpu_per_wall"],
                  rec["profile_distortion"], tr.summary()), flush=True)
        for name, share in rec["bucket_shares"].items():
            if share >= 0.005:
                print("          {:34s} {:5.1f}%".format(name, 100 * share), flush=True)
        if tr.throttled:
            print(" [warn] thermally clamped for {:.0f}%: judge on "
                  "cpu".format(100 * (tr.throttle_fraction or 0)), flush=True)
    print("\nwrote {}".format(out_path), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
