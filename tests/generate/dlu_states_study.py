"""DLU accuracy on state energies and splittings — the state-level record.

**What this decides.** Whether a `-DLU` Hamiltonian can be recommended for spectroscopy, and
what the runtime warning emitted at selection should say. The operator-level record
(the x2c validation record) bounds the Hamiltonian perturbation; nothing yet states what
that does to a *converged multireference spectrum*, which is the number a user quotes.

⚠ **Both sides run through the same code**: the exact reference is `x2c_approx="1e-dlu"` with
``partition="single"`` — the trivial one-fragment partition, for which DLU *is* the exact
molecular decoupling — against ``partition="atoms"``. Comparing against the ordinary ``"1e"``
route would fold the Kuiva-vs-PySCF implementation difference into the answer, and that
difference is the same size as the DLU error being measured (x2c validation record, section 1).

**Construction, stated with every number**: SA-CASSCF (``mode="second-order"``,
``conv_grad=1e-6``) over the stated roots, ``x2c-SVPall-2c``, ``screening="none"`` throughout —
this is the *one-electron* axis, and the two-electron screening is identical on both sides by
construction. The scalar SCF is ``sfx2c1e`` on both sides (the decoupling axis does not touch
it), so both runs start from identical orbitals and the converged difference is the operator
plus its orbital relaxation — exactly what a user of the `-DLU` Hamiltonian gets.

⚠ **Thresholds fixed before any number existed**, so the verdict cannot be argued into place:

* free-ion control (Ce3+): a single atom is one fragment, so ``atoms`` and ``single`` must
  agree to numerical noise — **1e-6 cm^-1** on every level. A control failure voids the study.
* ligand-field splittings: **0.1 cm^-1 absolute** (the bar the suite asserts free-ion
  degeneracies at) *and* **0.2 % relative** per level (a ligand-field band needs both a floor
  and a percentage).
* ground-block principal g values: **0.3 % relative** (the margin the validation record
  currently agrees with OpenMolcas at).

Below all three, DLU is recommendable for splittings and g values at this level and the
warning is rewritten to say so with the measured bound; above any, the warning stays "no
statement exists" with the measured counterexample.

Run:  ``python tests/generate/dlu_states_study.py [--only ticl3] [--out temp/dlu_states.json]``
One case per invocation stays comfortably inside the ten-minute rule; results are written
incrementally after every case.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from kuiva.interface import api                                          # noqa: E402
from kuiva.props.dump import property_matrices                           # noqa: E402
from tests.generate import systems                                       # noqa: E402
from tests.generate.progress import Heartbeat                            # noqa: E402

EH_TO_CM = 219474.6313632

THRESHOLDS = {
    "control_cm": 1e-6,
    "splitting_abs_cm": 0.1,
    "splitting_rel": 0.002,
    "g_rel": 0.003,
}

#: (key, n_states); cheapest first. ce3p is the control (one atom: atoms == single).
CASES = [("ce3p", 14), ("ticl3", 10), ("cecl3", 14)]


def one_side(system, n_states: int, partition: str, memory_gb: float) -> Dict[str, object]:
    """One full pipeline with the stated partition; returns levels and reductions."""
    t0, c0 = time.time(), time.process_time()
    mol = api.Molecule(atoms=system.atoms, basis=system.basis,
                       charge=system.charge, spin=system.spin)
    reference = api.spinor_reference(mol, screening="none", memory_gb=memory_gb,
                                     x2c_approx="1e-dlu",
                                     decoupling_options={"partition": partition})
    outcome = api.casscf(reference, character=(system.atoms[0][0], system.active_l),
                         n_active=2 * system.ncas, n_active_elec=system.nelecas,
                         n_states=n_states, mode="second-order", conv_grad=1e-6,
                         report=False)
    props = reference.data.properties
    tdm = outcome.ci.transition_densities()
    matrices = property_matrices(outcome.coeff, outcome.active.spaces, tdm,
                                 outcome.ci.total_energies, props, reference.data.s_ao)
    energies = np.asarray(outcome.ci.total_energies, dtype=float)
    blocks = []
    for b in matrices.analyse(tol_cm=1.0):
        blocks.append({"size": int(b.size), "energy_cm": float(b.energy_cm),
                       "spread_cm": float(b.spread_cm),
                       "g_values": [float(g) for g in b.g_values]})
    soc = reference.data.soc
    return {
        "partition": partition,
        "converged": bool(outcome.converged),
        "e_total": [float(e) for e in energies],
        "levels_cm": [float(x) for x in (energies - energies[0]) * EH_TO_CM],
        "blocks": blocks,
        "decoupling": soc.decoupling.provenance() if hasattr(soc.decoupling, "provenance")
                      else str(soc.decoupling),
        "wall_s": time.time() - t0, "cpu_s": time.process_time() - c0,
    }


def compare(exact: Dict, dlu: Dict) -> Dict[str, object]:
    lv_e = np.asarray(exact["levels_cm"])
    lv_d = np.asarray(dlu["levels_cm"])
    d_levels = lv_d - lv_e
    # relative on the splitting itself; the ground level (0) is excluded
    rel = [abs(d) / abs(e) for d, e in zip(d_levels[1:], lv_e[1:]) if abs(e) > 1e-6]
    g_rel = []
    for be, bd in zip(exact["blocks"], dlu["blocks"]):
        for ge, gd in zip(be["g_values"], bd["g_values"]):
            if abs(ge) > 1e-6:
                g_rel.append(abs(gd / ge - 1.0))
    return {
        "delta_total_ground_eh": dlu["e_total"][0] - exact["e_total"][0],
        "delta_levels_cm": [float(x) for x in d_levels],
        "worst_level_shift_cm": float(np.max(np.abs(d_levels))),
        "worst_level_shift_rel": float(max(rel)) if rel else 0.0,
        "worst_g_shift_rel": float(max(g_rel)) if g_rel else 0.0,
        "delta_spread_cm": [float(bd["spread_cm"] - be["spread_cm"])
                            for be, bd in zip(exact["blocks"], dlu["blocks"])],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="temp/dlu_states.json")
    ap.add_argument("--only", default="", help="comma-separated system keys")
    ap.add_argument("--memory-gb", type=float, default=4.0)
    args = ap.parse_args()
    only = {s for s in args.only.split(",") if s}

    data: Dict[str, object] = {"thresholds": THRESHOLDS, "records": {}}
    if os.path.exists(args.out):
        with open(args.out) as fh:
            data = json.load(fh)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    hb = Heartbeat("dlu_states", budget_seconds=45 * 60,
                   meta={"cases": [k for k, _ in CASES]})

    for i, (key, n_states) in enumerate(CASES):
        if only and key not in only:
            continue
        if hb.expired:
            hb.tick(i, stage="wall budget exhausted before {}".format(key))
            break
        system = systems.get(key)
        hb.tick(i, stage="{}: exact side (partition=single)".format(key))
        exact = one_side(system, n_states, "single", args.memory_gb)
        hb.tick(i, stage="{}: DLU side (partition=atoms)".format(key))
        dlu = one_side(system, n_states, "atoms", args.memory_gb)
        rec = {
            "label": system.label, "basis": system.basis, "n_states": n_states,
            "construction": "SA-CASSCF over {} roots, CAS({}, {} spinors), {}, "
                            "screening=none".format(n_states, system.nelecas,
                                                    2 * system.ncas, system.basis),
            "exact": exact, "dlu": dlu, "delta": compare(exact, dlu),
        }
        data["records"][key] = rec
        with open(args.out + ".tmp", "w") as fh:
            json.dump(data, fh, indent=1, sort_keys=True)
        os.replace(args.out + ".tmp", args.out)
        d = rec["delta"]
        print("{:8s} worst level shift {:.3e} cm^-1 ({:.2e} rel), worst g shift {:.2e} rel"
              .format(key, d["worst_level_shift_cm"], d["worst_level_shift_rel"],
                      d["worst_g_shift_rel"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
