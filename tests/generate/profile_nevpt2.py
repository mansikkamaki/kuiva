"""SC-NEVPT2 cost decomposition on a large-virtual-space system.

The oxygen profile (recorded) put 57% of the correction in the
shifted-space ``H_act`` sigma applications (GEMM-shaped — the closed sigma gate) and **3%**
in the single-external perturber assembly, with the recorded caveat that a large virtual
space inverts the ordering. This script measures that inversion, applying the C++ port gate's
gate as declared: the assembly becomes a port candidate only if it is serial
(`cpu/wall ~ 1` at `KMP_BLOCKTIME=0`), exceeds 25% of the correction's CPU seconds at the
largest ad-hoc-reachable size, and its share does not fall with size.

Instrument
----------
One ``sc_nevpt2`` per system (all-electron, all virtuals, defaults), read through the
module's own timer regions (`kuiva/util/timing.py` — wall *and* CPU per region, so each
region carries its own serial/threaded verdict). The size axis is the external-label count
``n_ext``: O -> Te -> TlH spans it at fixed instrument, and TlH in the project basis is the
"heavy molecular case" recorded as untested.

⚠ **References are CASCIs at the scalar SCF orbitals, and no energy from this script may be
quoted** — the notes record exactly why (``Sir`` measures the Brillouin
violation of an unconverged reference; on TlH it reaches -36.7 Eh). The *cost* is what is
measured here, and cost is fixed by the dimensions (`ndet`, `n_act`, `n_ext`) which a
converged reference would not change. One value-dependence exists and is recorded rather
than hidden: the small-norm cutoff skips less work when norms are inflated, so a bad
reference can only *overstate* the perturber-assembly share — conservative in exactly the
direction the gate needs.

Ambient BLAS is pinned to 4 threads (the dev-box thread cap), so a
GEMM-shaped region reads `cpu/wall > 1` and a serial one reads 1.0 — `KMP_BLOCKTIME=0`
keeps spin-wait from blurring that line.

Usage::

    python tests/generate/profile_nevpt2.py                 # O + Te + TlH, ~6 min
    python tests/generate/profile_nevpt2.py --only tlh      # one system (incremental)
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
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
if "numpy" in sys.modules:                                              # pragma: no cover
    raise SystemExit("numpy was imported before the thread pins could be set; run this "
                     "script directly")

import numpy as np                                                      # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import systems as sysdef                                                # noqa: E402
import thermal                                                          # noqa: E402
from nevpt2_frozen_core import resolve                               # noqa: E402

OUT = REPO / "temp/profile_nevpt2.json"

MEMORY_GB = 8.0

#: Region aggregation: a timer label containing the key belongs to the named group. The
#: same label can appear at several paths (once per class that triggered the lazily built
#: quantity); grouping is by label, summed over paths.
GROUPS = (
    ("single-external assembly", ("single-external",)),
    ("H_act sigma (shifted spaces)", ("H_act on the",)),
    ("ladder-string vector prep", ("excitation vectors", "pair-annihilated",
                                   "pair-created")),
    ("state RDMs", ("state RDMs",)),
    ("pseudo-canonicalization", ("pseudo-canonicalization",)),
)

SYSTEM_KEYS = ("o", "te", "tlh")


def run_system(system: sysdef.System) -> Dict:
    from kuiva.interface import api
    from kuiva.pt.nevpt2 import sc_nevpt2
    from kuiva.util import resources as res
    from kuiva.util import timing

    res.BUDGET.clear()
    molecule = api.Molecule(atoms=system.atoms, basis=system.basis, charge=system.charge,
                            spin=system.spin)
    # screening="none": a cost measurement; the two-electron picture change costs a
    # four-component atomic solve it cannot affect.
    reference = api.spinor_reference(molecule, memory_gb=MEMORY_GB, screening="none")
    n_inactive = reference.data.nelec_total - system.nelecas
    if n_inactive % 2:
        raise SystemExit("{}: odd inactive spinor count".format(system.key))
    selection = (dict(character=([0], system.active_l), n_active=2 * system.ncas,
                      n_active_elec=system.nelecas) if system.active_l else
                 dict(active=list(range(n_inactive, n_inactive + 2 * system.ncas)),
                      n_active_elec=system.nelecas))
    # ⚠ CASCI at the SCF orbitals — cost instrument only; see the module docstring.
    cas = api.casci(reference, n_states=system.soc_states, report=False, **selection)
    spaces = cas.spaces

    h_ao = reference.h_one_electron()
    timing.reset()
    t0, c0 = time.time(), time.process_time()
    corrected = sc_nevpt2(reference.factors, h_ao, cas.coeff, spaces, cas.vectors,
                          system.nelecas, energies=cas.energies,
                          e_nuc=reference.data.e_nuc, report=False)
    wall = time.time() - t0
    cpu = time.process_time() - c0

    nodes = [{"path": n.path, "label": n.label, "depth": n.depth, "calls": n.calls,
              "wall": round(n.wall, 4), "cpu": round(n.cpu, 4),
              "cpu_per_wall": round(n.parallel_ratio, 2)}
             for n in timing.REGISTRY.nodes()]
    groups: Dict[str, Dict[str, float]] = {}
    for name, keys in GROUPS:
        rows = [n for n in nodes if any(k in n["label"] for k in keys)]
        g_wall = sum(r["wall"] for r in rows)
        g_cpu = sum(r["cpu"] for r in rows)
        groups[name] = {"wall": round(g_wall, 3), "cpu": round(g_cpu, 3),
                        "cpu_share": round(g_cpu / cpu, 4) if cpu else 0.0,
                        "cpu_per_wall": round(g_cpu / g_wall, 2) if g_wall else 0.0}

    n_ext = int(spaces.n_inactive + spaces.n_virtual)
    return {"key": system.key, "label": system.label, "basis": system.basis,
            "n_states": int(system.soc_states), "nao": int(reference.data.nao),
            "nspinor": int(reference.nspinor), "n_act": int(2 * system.ncas),
            "n_inactive": int(spaces.n_inactive), "n_virtual": int(spaces.n_virtual),
            "n_ext": n_ext, "ndet": int(cas.vectors.shape[1]),
            "reference_note": "CASCI at scalar SCF orbitals; costs only, energies "
                              "not quotable (pt notes 4.2)",
            "e2_sanity": float(corrected.e2[0]),
            "wall": round(wall, 3), "cpu": round(cpu, 3),
            "cpu_per_wall": round(cpu / wall, 2) if wall else 0.0,
            "groups": groups, "regions": nodes}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--only", action="append", default=None,
                    help="system key (repeatable); default: {}".format(SYSTEM_KEYS))
    args = ap.parse_args(argv)

    os.environ.setdefault("KUIVA_MEMORY_GB", str(MEMORY_GB))

    out_path = Path(args.out)
    if out_path.exists():
        record = json.loads(out_path.read_text())
    else:
        record = {"schema": 1, "generator": "tests/generate/profile_nevpt2.py",
                  "ambient_blas_threads": 4,
                  "kmp_blocktime": os.environ.get("KMP_BLOCKTIME"),
                  "environment": thermal.describe_environment(), "records": []}

    keys = args.only or list(SYSTEM_KEYS)
    for key in keys:
        with thermal.track_resources() as tr:
            rec = run_system(resolve(key))
        rec["resources"] = tr.as_dict()
        record["records"] = [r for r in record["records"] if r["key"] != key]
        record["records"].append(rec)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(record, indent=2, sort_keys=True))
        print("[nevpt2] {:5s} n_ext={:3d} ndet={:5d} | wall {:7.2f} s  cpu {:7.2f} s "
              "(cpu/wall {:4.2f}) | {}".format(
                  rec["key"], rec["n_ext"], rec["ndet"], rec["wall"], rec["cpu"],
                  rec["cpu_per_wall"], tr.summary()), flush=True)
        for name, g in rec["groups"].items():
            print("          {:30s} {:5.1f}% of cpu  (cpu/wall {:4.2f})".format(
                name, 100 * g["cpu_share"], g["cpu_per_wall"]), flush=True)
        if tr.throttled:
            print(" [warn] thermally clamped for {:.0f}%: judge on "
                  "cpu".format(100 * (tr.throttle_fraction or 0)), flush=True)
    print("\nwrote {}".format(out_path), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
