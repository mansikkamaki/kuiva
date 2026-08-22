"""Measure the stored-vs-integral-direct Cholesky cost crossover (development study).

The two fitting routes differ **only** in how the two-electron integrals reach the
decomposition: the stored route materializes the 8-fold packed array (``mol.intor('int2e',
aosym='s8')``, O(nao^4)) and gathers columns from it, the direct route evaluates shell-pair
batches as the pivoting asks for them (O(naux * nao^2)).  Everything else — the SCF, the
spin-orbit ingestion, every downstream consumer — is identical, so the route that is cheaper
on this phase is cheaper overall, and this script times exactly this phase and nothing else.

One measurement per invocation, appended to a JSON file immediately (results are written
incrementally, never only at the end).  Usage::

    python direct_crossover_study.py --system ti2cl6 --basis mixed-ti-tzvp --route stored
    python direct_crossover_study.py --report

Costs are judged on **CPU seconds** (the development box throttles thermally, so wall time
there is partly a temperature measurement); wall time is recorded beside it.  The stored
route needs the packed array plus the factors in RAM, which on a 16 GB machine limits it to
nao ~ 300; the direct route is also measured beyond that point, where the stored cost is
extrapolated from its measured components (the intor cost is a clean quartic and the
stored-column decomposition a clean cubic, both fitted from the measured points).
"""

from __future__ import print_function

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

BASES = {
    "svp": "x2c-SVPall-2c",
    "tzvp": "x2c-TZVPall-2c",
    "mixed-ti-tzvp": {"Ti": "x2c-TZVPall-2c", "Cl": "x2c-SVPall-2c"},
    "mixed-cl-tzvp": {"Ti": "x2c-SVPall-2c", "Cl": "x2c-TZVPall-2c"},
}

DEFAULT_OUT = os.path.join(ROOT, "temp", "direct_crossover.json")


def _load(path):
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return {"records": []}


def _save(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=1, sort_keys=True)
    os.replace(tmp, path)


def measure(system_key, basis_tag, route):
    from systems import get
    from kuiva.interface.api import Molecule
    from kuiva.interface.pyscf_bridge import (build_mole, _direct_cholesky, ao_layout)
    from kuiva.integrals.transform import (ThreeIndexAO, shell_pair_orbits,
                                           DEFAULT_CHOLESKY_TOL)
    import kuiva.util.resources as res

    res.ensure_configured(None)   # the configured site default, exactly as a run would see it

    s = get(system_key)
    mol = build_mole(Molecule(atoms=s.atoms, basis=BASES[basis_tag], charge=s.charge,
                              spin=s.spin), verbose=0)
    rec = {
        "system": system_key, "basis": basis_tag, "route": route,
        "nao": int(mol.nao), "nbas": int(mol.nbas),
        "threads": os.environ.get("KUIVA_NUM_THREADS", ""),
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    layout = ao_layout(mol)
    orbits = shell_pair_orbits(layout.ao_shell, layout.ao_atom, one_centre=True)

    if route == "stored":
        w0, c0 = time.perf_counter(), time.process_time()
        eri = mol.intor("int2e", aosym="s8")
        w1, c1 = time.perf_counter(), time.process_time()
        factors = ThreeIndexAO.from_eri(eri, int(mol.nao), DEFAULT_CHOLESKY_TOL,
                                        orbits=orbits, report=False)
        w2, c2 = time.perf_counter(), time.process_time()
        rec.update(intor_wall=w1 - w0, intor_cpu=c1 - c0,
                   chol_wall=w2 - w1, chol_cpu=c2 - c1,
                   total_wall=w2 - w0, total_cpu=c2 - c0,
                   naux=int(factors.naux), eri_gb=eri.nbytes / 2.0 ** 30)
    elif route == "direct":
        w0, c0 = time.perf_counter(), time.process_time()
        factors = _direct_cholesky(mol, tol=DEFAULT_CHOLESKY_TOL, report=False)
        w1, c1 = time.perf_counter(), time.process_time()
        rec.update(total_wall=w1 - w0, total_cpu=c1 - c0, naux=int(factors.naux))
    else:
        raise SystemExit("route must be 'stored' or 'direct'")
    return rec


def report(data):
    rows = {}
    for r in data["records"]:
        key = (r["nao"], r["system"], r["basis"])
        rows.setdefault(key, {}).setdefault(r["route"], []).append(r)
    print("%5s  %-24s  %6s  %12s  %12s  %8s" % (
        "nao", "system/basis", "naux", "stored cpu-s", "direct cpu-s", "direct/stored"))
    for key in sorted(rows):
        nao, system, basis = key
        by_route = rows[key]
        # several repeats: take the minimum CPU (least perturbed by the rest of the machine)
        stored = min((r["total_cpu"] for r in by_route.get("stored", [])), default=None)
        direct = min((r["total_cpu"] for r in by_route.get("direct", [])), default=None)
        naux = by_route.get("direct", by_route.get("stored"))[0].get("naux", "")
        ratio = ("%8.3f" % (direct / stored)) if stored and direct else "        "
        print("%5d  %-24s  %6s  %12s  %12s  %s" % (
            nao, "%s/%s" % (system, basis), naux,
            "%.1f" % stored if stored is not None else "-",
            "%.1f" % direct if direct is not None else "-", ratio))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--system", default=None)
    ap.add_argument("--basis", default=None, choices=sorted(BASES))
    ap.add_argument("--route", default=None, choices=["stored", "direct"])
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    data = _load(args.out)
    if args.report:
        report(data)
        return
    if not (args.system and args.basis and args.route):
        raise SystemExit("--system, --basis and --route are all required to measure")
    rec = measure(args.system, args.basis, args.route)
    data["records"].append(rec)
    _save(args.out, data)
    print(json.dumps(rec, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
