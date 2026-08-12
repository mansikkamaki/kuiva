"""Thread sweep on one BLAS-bound and one kernel-bound stage.

What this measures, and why the two stages are the point
--------------------------------------------------------
The thread budget is one number, but the two halves of this code want it spent in
opposite places, and that claim is what the region policy of :mod:`kuiva.util.threads`
rests on. So this script sweeps the budget over both:

* **BLAS-bound** — ``transform_3c`` on a real spinor reference (TiCl3, 210 spinors, 1139
  Cholesky vectors): a pure ``zgemm`` stage under
  :func:`kuiva.util.threads.blas_region`. Wall time must fall with the budget and
  ``cpu/wall`` must rise towards it.
* **Kernel-bound** — a two-site DMRG sweep under
  :func:`kuiva.util.threads.kernel_region`, measured against the *ambient* policy it
  replaced (MKL and the compiled kernels both at the budget). The claim is not that the
  clamp is faster in wall time; it is that the threaded BLAS in this layer buys nothing
  and is charged for it, so clamping it should hold wall time and give back CPU seconds
  — and CPU seconds are what the port gate ranks by.

⚠ Every number here is a **ratio within one configuration**, never an absolute rate: the
development box thermally throttles, so absolutes drift with the package
temperature while ratios measured back to back do not. Run it on an idle, cool machine.

Bounded within the ten-minute ad-hoc budget: a fixed sweep count and a fixed configuration list, well under ten
minutes in total, with each row written to the JSON output the moment it completes.

Usage::

    KMP_BLOCKTIME=0 python tests/generate/bench_threads.py
    KMP_BLOCKTIME=0 python tests/generate/bench_threads.py --stage blas
"""
from __future__ import annotations

# ⚠ Imported before numpy on purpose: this module pins the ambient OpenMP/MKL widths at
# import and refuses to run if numpy beat it to it. Everything below controls the widths
# at *run time* instead (MKL_Set_Num_Threads / _Local through kuiva.util.threads), which
# is what makes a sweep inside one process possible at all.
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import profile_dmrg_sweep as dmrg_profile                              # noqa: E402

import numpy as np                                                     # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

OUT = REPO / "temp/bench_threads.json"

#: Thread budgets to sweep. 4 is the pinned cap on this box; 8 would measure hyperthreads
#: competing with the desktop, which is a fact about the box and not about the code.
BUDGETS = (1, 2, 4)

#: The kernel-bound configuration: the measured fat-node shape, two sweeps, two roots.
DMRG_CONFIG = ("n=12 fat, D=32", 12, 6, 3, 32)

#: The BLAS-bound stage: TiCl3 from the one source of truth for systems,
#: transformed bra=ket over the whole spinor space — 1.5 s of pure zgemm serial, which is
#: enough that the threading is measured rather than the threading overhead.
BLAS_SYSTEM_KEY = "ticl3"
BLAS_REPEATS = 3


def _timed(fn):
    wall, cpu = time.perf_counter(), time.process_time()
    value = fn()
    return value, time.perf_counter() - wall, time.process_time() - cpu


# --- the BLAS-bound stage ------------------------------------------------------------------


def _blas_setup():
    from systems import SYSTEMS_BY_KEY

    from kuiva.interface.api import Molecule, spinor_reference
    from kuiva.util.logging import set_verbosity

    set_verbosity("ERROR")
    system = SYSTEMS_BY_KEY[BLAS_SYSTEM_KEY]
    mol = Molecule(system.atoms, basis=system.basis, charge=system.charge,
                   spin=system.spin)
    # ⚠ screening="none": this benchmark times a GEMM stage, and the two-electron
    # picture change changes no timing while costing a four-component atomic solve per
    # element. A caller that does not care about SOC says so.
    ref = spinor_reference(mol, screening="none")
    return ref.factors, ref.spinors_in_ao()


def bench_blas(budgets: Sequence[int]) -> List[Dict]:
    from kuiva.integrals.transform import transform_3c
    from kuiva.util import threads

    factors, c = _blas_setup()
    transform_3c(factors, c, c)                       # warm the dispatch and the pages
    rows = []
    for n in budgets:
        threads.set_budget(n)
        best = None
        for _ in range(BLAS_REPEATS):
            with threads.blas_region():
                block, wall, cpu = _timed(lambda: transform_3c(factors, c, c))
            if best is None or wall < best[0]:
                best = (wall, cpu, float(np.linalg.norm(block)))
        wall, cpu, norm = best
        rows.append({"stage": "blas", "budget": n, "wall": round(wall, 3),
                     "cpu": round(cpu, 3), "cpu_per_wall": round(cpu / wall, 2),
                     "check_norm": norm,
                     "nao": int(factors.nao), "naux": int(factors.naux),
                     "nspinor": int(c.shape[1])})
        print(json.dumps(rows[-1]), flush=True)
    return rows


# --- the kernel-bound stage ----------------------------------------------------------------


def bench_kernel(budgets: Sequence[int]) -> List[Dict]:
    """The DMRG sweep under both policies, so the clamp is measured against what it replaced."""
    from kuiva.ci import kernels
    from kuiva.util import threads

    label, n, k, modes, max_bond = DMRG_CONFIG
    previous = kernels.set_preferred_backend("native")
    mkl = threads._mkl()
    ambient = None if mkl is None else mkl.max_threads()
    rows = []
    try:
        for n_threads in budgets:
            threads.set_budget(n_threads)
            for policy in ("ambient", "kernel_region"):
                # "ambient" reproduces the pre-region behaviour exactly: the BLAS *and*
                # the compiled kernels both at the budget, nobody clamped.
                op, state = dmrg_profile._setup(n, k, modes, max_bond)
                if policy == "ambient" and mkl is not None:
                    mkl.set_global(n_threads)
                result, wall, cpu = _timed(lambda: dmrg_profile._solve(op, state, max_bond))
                if mkl is not None:
                    mkl.set_global(ambient)
                rows.append({"stage": "kernel", "config": label, "budget": n_threads,
                             "policy": policy, "wall": round(wall, 3),
                             "cpu": round(cpu, 3), "cpu_per_wall": round(cpu / wall, 2),
                             "energy_sa": float(np.mean(result.energies))})
                print(json.dumps(rows[-1]), flush=True)
    finally:
        kernels.set_preferred_backend(previous)
        if mkl is not None and ambient is not None:
            mkl.set_global(ambient)
        threads._reset_cache()
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--stage", choices=("blas", "kernel", "both"), default="both")
    args = ap.parse_args(argv)

    rows: List[Dict] = []
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.stage in ("blas", "both"):
        rows += bench_blas(BUDGETS)
        out.write_text(json.dumps(rows, indent=1))
    if args.stage in ("kernel", "both"):
        rows += bench_kernel(BUDGETS)
        out.write_text(json.dumps(rows, indent=1))
    print("wrote {}".format(out))

    print("\n stage   budget  policy         wall [s]   cpu [s]  cpu/wall")
    print(" ------  ------  -------------  --------  --------  --------")
    for row in rows:
        print(" {:6s}  {:6d}  {:13s}  {:8.3f}  {:8.3f}  {:8.2f}".format(
            row["stage"], row["budget"], row.get("policy", "blas_region"),
            row["wall"], row["cpu"], row["cpu_per_wall"]))
    print("\n⚠ ratios within a configuration only; absolutes drift with the package "
          "temperature.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
