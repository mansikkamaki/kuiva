"""Are the Loewdin atomic charges worth keeping? — the characterization that decides.

**What this decides.** The Loewdin *charges* are recorded as the weakest number the props
package produces — sign-opposite to Mulliken on TiCl3 (the props validation record,
section 6) — while the reduced *orbital* populations they share machinery with are robust.
Three outcomes were named when the question was posed: keep the charge with a sharper warning,
replace it with a better-conditioned partition, or withdraw it from the public surface while
keeping the orbital populations. This script produces the numbers that pick one.

**Design.** For every (system, basis) pair: one scalar ``sfx2c1e`` SCF, then the Mulliken and
Loewdin atomic charges of the *same converged density* —

    Mulliken:  q_A = Z_A - sum_{mu in A} (D S)_{mu mu}
    Loewdin:   q_A = Z_A - sum_{mu in A} (S^1/2 D S^1/2)_{mu mu}

so every difference is the partition and nothing else. Two probes, matched to the two ways a
charge can fail:

* **Partition disagreement**: |q_L - q_M| per atom, across chemically different systems
  (ionic Ti/Ce halides, the polar covalent TlH and HI).
* **Basis sensitivity**: q(TZVP) - q(SVP) per scheme. A charge that moves by a large fraction
  of itself when the basis improves is not a property of the molecule. This is the
  better-conditioned test, because it needs no opinion about which partition is "right".

The usual explanation for Loewdin's drift is diffuse functions (S^1/2 spreads a diffuse tail's
orthogonalization over every centre it overlaps), so the smallest primitive exponent per atom
is recorded with every row and the SVP -> TZVP step (which adds diffuse primitives) is the
test of that prediction.

Everything is front-end only (``with_soc=False``, no screening), so the whole matrix runs in
about a minute; results are rewritten after every pair.

Run:  ``python tests/generate/lowdin_charge_study.py [--out temp/lowdin_charges.json]``
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
from scipy.linalg import sqrtm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from kuiva.interface import api                                          # noqa: E402
from kuiva.interface.pyscf_bridge import run_scalar_x2c                  # noqa: E402
from tests.generate import systems                                       # noqa: E402

SYSTEMS = ["ticl3", "tif3", "cecl3", "tlh", "hi"]
BASES = ["x2c-SVPall-2c", "x2c-TZVPall-2c"]


def charges(data) -> Dict[str, List[float]]:
    layout = data.ao_layout
    s = np.asarray(data.s_ao)
    # Restricted/ROHF density; every system here uses one MO set.
    (c,) = data.mo_sets()
    d = (c * data.mo_occ) @ c.T
    s_half = np.real(sqrtm(s))
    per_ao_mull = np.einsum("ij,ji->i", d, s)
    per_ao_loew = np.diag(s_half @ d @ s_half)
    z = np.asarray(layout.atom_charges, dtype=float)
    atom = np.asarray(layout.ao_atom)
    natm = int(atom.max()) + 1
    q_m = z - np.array([per_ao_mull[atom == a].sum() for a in range(natm)])
    q_l = z - np.array([per_ao_loew[atom == a].sum() for a in range(natm)])
    # smallest primitive exponent on each atom: the diffuseness the explanation blames
    min_exp = [min(min(sh.exponents) for sh in layout.shells if sh.atom == a)
               for a in range(natm)]
    return {"symbols": [str(sym) for sym in layout.atom_symbols],
            "mulliken": [float(x) for x in q_m],
            "lowdin": [float(x) for x in q_l],
            "min_exponent": [float(x) for x in min_exp]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="temp/lowdin_charges.json")
    ap.add_argument("--memory-gb", type=float, default=4.0)
    ap.add_argument("--only", default="", help="comma-separated system keys")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    only = {s for s in args.only.split(",") if s}

    data: Dict[str, object] = {"records": {}}
    if os.path.exists(args.out):
        with open(args.out) as fh:
            data = json.load(fh)
    for key in SYSTEMS:
        if only and key not in only:
            continue
        system = systems.get(key)
        for basis in BASES:
            if "{}/{}".format(key, basis) in data["records"]:
                continue
            t0 = time.time()
            mol = api.Molecule(atoms=system.atoms, basis=basis,
                               charge=system.charge, spin=system.spin)
            scf = run_scalar_x2c(mol, with_soc=False, memory_gb=args.memory_gb, verbose=0)
            rec = charges(scf)
            rec.update(converged=bool(scf.converged), e_scf=float(scf.e_scf),
                       wall_s=time.time() - t0)
            data["records"]["{}/{}".format(key, basis)] = rec
            with open(args.out + ".tmp", "w") as fh:
                json.dump(data, fh, indent=1, sort_keys=True)
            os.replace(args.out + ".tmp", args.out)
            print("{:8s} {:16s} conv={} q_M={} q_L={}".format(
                key, basis, rec["converged"],
                " ".join("{:+.2f}".format(q) for q in rec["mulliken"]),
                " ".join("{:+.2f}".format(q) for q in rec["lowdin"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
