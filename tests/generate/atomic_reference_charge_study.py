"""Validation battery for the atomic-reference charges, on Kuiva's own implementation.

The scheme was chosen by a prototype battery (the props validation record tells the
story); this script re-runs that battery through Kuiva's shipped implementation — the
spherically constrained ``sfx2c1e`` AOC reference with the atomic mean field's per-element
default states (neutral atom; trivalent ion on the f block) — so the committed numbers
describe the code users run, not the prototype. Criteria, fixed by the Loewdin failure:

* sign sanity on the five systems (metal positive, H positive in HI);
* basis drift SVP -> TZVP well under Mulliken's 0.45 e (target ~0.1 e);
* a *correlated* density moves a charge smoothly, not qualitatively: the TiCl3 charge from
  the SA-CASSCF spin-traced 1-RDM against the SCF density on the same run.

Run:  ``python tests/generate/atomic_reference_charge_study.py [--out temp/atomic_reference_charges.json]``
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from kuiva.interface import api                                          # noqa: E402
import kuiva.util.resources as res                                       # noqa: E402
from tests.generate import systems                                       # noqa: E402

SYSTEMS = ["ticl3", "tif3", "cecl3", "tlh", "hi"]
BASES = ["x2c-SVPall-2c", "x2c-TZVPall-2c"]


def scf_charges(key: str, basis: str, memory_gb: float) -> Dict[str, object]:
    s0 = systems.get(key)
    t0 = time.time()
    res.clear()
    mol = api.Molecule(atoms=s0.atoms, basis=basis, charge=s0.charge, spin=s0.spin)
    ref = api.spinor_reference(mol, screening="none", memory_gb=memory_gb,
                               atomic_reference=True)
    q = ref.atomic_reference_charges(report=False)
    return {"system": key, "basis": basis,
            "q": [round(float(x), 4) for x in q.charge],
            "configurations": q.configurations, "wall_s": round(time.time() - t0, 1)}


def casscf_charges(key: str, basis: str, memory_gb: float) -> Dict[str, object]:
    """The correlated-density check: SCF-guess charges vs SA-CASSCF-density charges."""
    s0 = systems.get(key)
    res.clear()
    mol = api.Molecule(atoms=s0.atoms, basis=basis, charge=s0.charge, spin=s0.spin)
    ref = api.spinor_reference(mol, screening="none", memory_gb=memory_gb,
                               atomic_reference=True)
    q_scf = ref.atomic_reference_charges(report=False)
    outcome = api.casscf(ref, character=(s0.atoms[0][0], s0.active_l),
                         n_active=2 * s0.ncas, n_active_elec=s0.nelecas,
                         n_states=2, report=False)
    # state-averaged 1-RDM over ALL spinors: inactive occupied + the CI's state-averaged
    # active density, assembled in the spinor basis (spinor occupations are 0/1).
    spaces = outcome.active.spaces
    n = outcome.coeff.shape[1]
    gamma = np.zeros((n, n), dtype=complex)
    inact = np.asarray(spaces.inactive)
    gamma[inact[:, None], inact] = np.eye(inact.size)
    act = np.asarray(spaces.active)
    gamma[act[:, None], act] = outcome.ci.gamma
    q_cas = ref.atomic_reference_charges(coeff=outcome.coeff, dm=gamma, report=False)
    return {"system": key, "basis": basis,
            "q_scf": [round(float(x), 4) for x in q_scf.charge],
            "q_casscf": [round(float(x), 4) for x in q_cas.charge],
            "converged": bool(outcome.converged)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="temp/atomic_reference_charges.json")
    ap.add_argument("--memory-gb", type=float, default=6.0)
    ap.add_argument("--skip-casscf", action="store_true")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    data: Dict[str, object] = {"battery": [], "casscf": None}
    for key in SYSTEMS:
        for basis in BASES:
            row = scf_charges(key, basis, args.memory_gb)
            data["battery"].append(row)
            with open(args.out, "w") as fh:
                json.dump(data, fh, indent=1, sort_keys=True)
            print("%-6s %-16s q = %s   (%.0f s)" % (
                key, basis, " ".join("%+.3f" % x for x in row["q"]), row["wall_s"]))
    if not args.skip_casscf:
        row = casscf_charges("ticl3", "x2c-SVPall-2c", args.memory_gb)
        data["casscf"] = row
        with open(args.out, "w") as fh:
            json.dump(data, fh, indent=1, sort_keys=True)
        print("casscf check: q_scf %s -> q_casscf %s" % (
            " ".join("%+.3f" % x for x in row["q_scf"]),
            " ".join("%+.3f" % x for x in row["q_casscf"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
