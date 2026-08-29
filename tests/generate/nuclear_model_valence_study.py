"""What a finite nucleus does to a VALENCE property.

**What this decides.** Whether a heavy-element g factor or splitting quoted from Kuiva needs
`nuclear_model="gaussian"`, or whether the `"point"` default is defensible up to the bar those
numbers are held to. It also decides whether the Tier-2 DIRAC references should be regenerated
against DIRAC's own default (a Gaussian nucleus) instead of pinning DIRAC to a point one.

`nuclear_model="gaussian"` landed with its effect measured on the **core**: operator norms and
the deep 2p j-splitting of one atom, growing from 5.6e-08 (Ne) to 2.8e-03 (Hg) relative. Those
are the quantities that are cheap and that make the Z-trend unambiguous, and they bound
**nothing** a user actually quotes — a g factor, a ligand-field splitting and a CASSCF spectrum
are valence quantities, and a large core shift neither implies nor excludes a valence one.

⚠ **Both sides run through the same code and differ in one argument.** Everything else — basis,
active space, state count, optimizer settings, screening — is identical by construction, so the
converged difference is the nuclear model plus its orbital relaxation, which is exactly what a
user switching the option gets.

⚠ **The default screening (`x2camf`) is on, deliberately.** The nuclear model is in the atomic
mean field's cache key, so the honest measurement pays a second four-component atomic solve per
element; running `screening="none"` would measure only the one-electron channel and would not
be the number a user sees. The f-block half is the expensive one and is gated behind
`--include-lanthanide`.

The cases, in increasing cost:

* the **np¹ series** (B, Al, Ga, In, Tl; Z = 5 → 81) — the valence counterpart of the core
  Z-trend, one valence p electron all the way down, so the ²P splitting and the analytic Landé
  g values 2/3 and 4/3 are the whole content and nothing else varies with Z.
* **Ce(3+)** 4f¹ — a free-ion valence f case with an analytic g(²F5/2) = 6/7.
* **TiCl3** — the 3d complex: a real ligand-field splitting on top of SOC.
* **CeCl3** — the lanthanide complex (gated).

Run::

    python tests/generate/nuclear_model_valence_study.py --only b,al,ga
    python tests/generate/nuclear_model_valence_study.py --only ticl3
    python tests/generate/nuclear_model_valence_study.py --only cecl3 --include-lanthanide

One or two cases per invocation stays inside the ten-minute rule; records are written
incrementally after every case.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import systems as sysdef                                                     # noqa: E402

from kuiva.interface import api                                              # noqa: E402
from kuiva.props.dump import property_matrices                               # noqa: E402
from kuiva.util import resources as res                                      # noqa: E402
from progress import Heartbeat                                               # noqa: E402

EH_TO_CM = 219474.6313632
WALL_BUDGET_S = 9.5 * 60

#: (key, n_states, is_lanthanide).
#:
#: ⚠ The valence-shell selection is carried by each system's ``active_skip_pairs`` and
#: applied by ``systems.character_selection``; every splitting below is checked against
#: experiment, which is the only cheap guard that catches a core-shell selection.
CASES = (
    ("b",  6, False),
    ("al",  6, False),
    ("ga",  6, False),
    ("in",  6, False),
    ("tl",  6, False),
    ("ce3p", 14, True),
    ("ticl3", 10, False),
    ("cecl3", 14, True),
)

#: The check that the ordinal window named the valence shell and not a core one; the numbers
#: live beside the systems, not here.
EXPERIMENTAL_SPLITTING_CM = sysdef.EXPERIMENTAL_SPLITTING_CM

#: Bars the verdict is read against, fixed before any number existed.
#: ⚠ The g bar is the margin the property record agrees with OpenMolcas at; the splitting bars
#: are the suite's own free-ion degeneracy floor and a ligand-field percentage. A shift under
#: all of these is one no consumer of a Kuiva number could have noticed.
THRESHOLDS = {"g_rel": 0.003, "splitting_abs_cm": 0.1, "splitting_rel": 0.002}

#: ⚠ Below this a principal g is treated as **zero** and only its absolute shift is quoted.
#: A ligand field leaves transverse g values that vanish by symmetry, and a percentage taken
#: against one of them says nothing about anything: TiCl3's excited doublets sit at ~5e-07,
#: where a 4e-08 shift reads as "3.9e-02 relative". The project rule this instantiates: a
#: ligand-field band needs an absolute floor as well as a percentage.
G_FLOOR = 1e-3

#: ⚠ Below this (cm⁻¹) a level separation is **inside a degenerate manifold**, not a splitting,
#: and no percentage may be taken against it. Equal to the ``tol_cm`` the block analysis groups
#: at, so the two agree on what "one level" means.
LEVEL_FLOOR_CM = 1.0


def _system(key: str) -> sysdef.System:
    for s in sysdef.SYSTEMS:
        if s.key == key:
            return s
    raise SystemExit("no system {!r}".format(key))


def one_side(system, n_states: int, nuclear_model: str, *, memory_gb: float,
             screening: str) -> Dict[str, object]:
    """One full pipeline with the stated nuclear model; returns levels and reductions."""
    t0, c0 = time.time(), time.process_time()
    mol = api.Molecule(atoms=system.atoms, basis=system.basis, charge=system.charge,
                       spin=system.spin, nuclear_model=nuclear_model)
    reference = api.spinor_reference(mol, screening=screening, memory_gb=memory_gb)
    t_ref = time.time() - t0
    # ⚠ Through the shared helper, which is the one place the ordinal window is applied.
    outcome = api.casscf(reference, n_states=n_states, mode="second-order", conv_grad=1e-6,
                         report=False, **sysdef.character_selection(system))
    props = reference.data.properties
    tdm = outcome.ci.transition_densities()
    matrices = property_matrices(outcome.coeff, outcome.active.spaces, tdm,
                                 outcome.ci.total_energies, props, reference.data.s_ao)
    energies = np.asarray(outcome.ci.total_energies, dtype=float)
    blocks = [{"size": int(b.size), "energy_cm": float(b.energy_cm),
               "spread_cm": float(b.spread_cm),
               "g_values": [float(g) for g in b.g_values]}
              for b in matrices.analyse(tol_cm=1.0)]
    soc = reference.data.soc
    return {
        "nuclear_model": nuclear_model,
        "active_space": str(outcome.active.description),
        "converged": bool(outcome.converged),
        "e_total": [float(e) for e in energies],
        "levels_cm": [float(x) for x in (energies - energies[0]) * EH_TO_CM],
        "blocks": blocks,
        "provenance": soc.provenance() if hasattr(soc, "provenance") else str(soc),
        "reference_wall_s": round(t_ref, 2),
        "wall_s": round(time.time() - t0, 2), "cpu_s": round(time.process_time() - c0, 2),
    }


def compare(point: Dict, gaussian: Dict) -> Dict[str, object]:
    """Point → gaussian shifts, through phase-invariant quantities only."""
    lv_p = np.asarray(point["levels_cm"])
    lv_g = np.asarray(gaussian["levels_cm"])
    d = lv_g - lv_p
    # ⚠ **Same trap as the g floor, one level down.** Dividing a level shift by a level that
    # sits INSIDE a degenerate manifold is dividing by that manifold's residual splitting:
    # Ce(3+)'s ²F5/2 members sit ~2e-03 cm⁻¹ apart, and a 1.4e-02 cm⁻¹ shift against one of
    # those reads as "6.3e-01 relative". The floor is LEVEL_FLOOR_CM, the same 1 cm⁻¹ the
    # block analysis groups at, so a denominator is only ever a real splitting. The quantity
    # to quote is `worst_block_gap_rel` below, which is per-BLOCK and cannot have this defect.
    rel = [abs(x) / abs(e) for x, e in zip(d[1:], lv_p[1:]) if abs(e) > LEVEL_FLOOR_CM]
    # ⚠ **A relative g shift needs an absolute floor.** A ligand field leaves transverse g
    # values that are ZERO by symmetry (TiCl3's excited doublets sit at ~5e-07), and dividing
    # by one of those turns a 4e-08 shift into "3.9e-02 relative". The floor below is the
    # threshold under which a g is treated as zero and only the absolute shift is quoted.
    g_abs, g_rel = [], []
    g_abs_ground, g_rel_ground = [], []
    for i, (bp, bg) in enumerate(zip(point["blocks"], gaussian["blocks"])):
        for gp, gg in zip(bp["g_values"], bg["g_values"]):
            g_abs.append(abs(gg - gp))
            if abs(gp) > G_FLOOR:
                g_rel.append(abs(gg / gp - 1.0))
            if i == 0:
                g_abs_ground.append(abs(gg - gp))
                if abs(gp) > G_FLOOR:
                    g_rel_ground.append(abs(gg / gp - 1.0))
    # the block-to-block gap is the "free-ion-like SOC gap" for an atom and the
    # ligand-field pattern for a complex; both are read off the same block energies
    e_p = np.asarray([b["energy_cm"] for b in point["blocks"]])
    e_g = np.asarray([b["energy_cm"] for b in gaussian["blocks"]])
    n = min(e_p.size, e_g.size)
    gap_rel = [abs(e_g[i] / e_p[i] - 1.0) for i in range(1, n) if abs(e_p[i]) > 1e-6]
    return {
        "delta_total_ground_eh": gaussian["e_total"][0] - point["e_total"][0],
        "delta_levels_cm": [float(x) for x in d],
        "worst_level_shift_cm": float(np.max(np.abs(d))),
        "worst_level_shift_rel": float(max(rel)) if rel else 0.0,
        "block_energies_point_cm": [float(x) for x in e_p],
        "block_energies_gaussian_cm": [float(x) for x in e_g],
        "worst_block_gap_rel": float(max(gap_rel)) if gap_rel else 0.0,
        "ground_g_point": point["blocks"][0]["g_values"],
        "ground_g_gaussian": gaussian["blocks"][0]["g_values"],
        "worst_g_shift_abs": float(max(g_abs)) if g_abs else 0.0,
        "worst_g_shift_rel": float(max(g_rel)) if g_rel else 0.0,
        "g_floor": G_FLOOR,
        "ground_g_shift_abs": float(max(g_abs_ground)) if g_abs_ground else 0.0,
        "ground_g_shift_rel": float(max(g_rel_ground)) if g_rel_ground else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="temp/nuclear_model_valence.json")
    ap.add_argument("--only", default="", help="comma-separated system keys")
    ap.add_argument("--memory-gb", type=float, default=6.0)
    ap.add_argument("--screening", default="x2camf",
                    help="⚠ the default is the point of the measurement; 'none' measures "
                         "only the one-electron channel and is not what a user gets")
    ap.add_argument("--wall-budget-min", type=float, default=9.5,
                    help="⚠ over 10 minutes is a user decision, not a default: the heavy "
                         "members need a four-component atomic solve per nuclear model and "
                         "In already exceeds one invocation of the ten-minute rule")
    ap.add_argument("--include-lanthanide", action="store_true",
                    help="run the f-block cases (a four-component atomic solve per nuclear "
                         "model; tens of minutes each)")
    args = ap.parse_args()
    only = {s for s in args.only.split(",") if s}

    data: Dict[str, object] = {"thresholds": THRESHOLDS, "screening": args.screening,
                               "records": {}}
    if os.path.exists(args.out):
        with open(args.out) as fh:
            data = json.load(fh)
    data.setdefault("records", {})
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    hb = Heartbeat("nuclear_model_valence", budget_seconds=args.wall_budget_min * 60,
                   meta={"cases": [c[0] for c in CASES]})
    for i, (key, n_states, lanth) in enumerate(CASES):
        if only and key not in only:
            continue
        if lanth and not args.include_lanthanide:
            print("  {} skipped (f block; pass --include-lanthanide)".format(key))
            continue
        if hb.expired:
            hb.tick(i, stage="wall budget exhausted before {}".format(key))
            print("wall budget exhausted before {}".format(key))
            break
        system = _system(key)
        hb.tick(i, stage="{}: point".format(key))
        print("  {} [{} states]".format(key, n_states), flush=True)
        point = one_side(system, n_states, "point", memory_gb=args.memory_gb,
                         screening=args.screening)
        hb.tick(i, stage="{}: gaussian".format(key))
        gaussian = one_side(system, n_states, "gaussian", memory_gb=args.memory_gb,
                            screening=args.screening)
        rec = {"label": system.label, "basis": system.basis, "n_states": n_states,
               "screening": args.screening,
               "skip_pairs": int(system.active_skip_pairs),
               "experimental_splitting_cm": EXPERIMENTAL_SPLITTING_CM.get(key),
               "construction": "SA-CASSCF over {} roots, CAS({}, {} spinors), {}, "
                               "screening={}".format(n_states, system.nelecas,
                                                     2 * system.ncas, system.basis,
                                                     args.screening),
               "point": point, "gaussian": gaussian,
               "delta": compare(point, gaussian)}
        data["records"][key] = rec
        with open(args.out + ".tmp", "w") as fh:
            json.dump(data, fh, indent=1, sort_keys=True)
        os.replace(args.out + ".tmp", args.out)
        d = rec["delta"]
        exp = EXPERIMENTAL_SPLITTING_CM.get(key)
        if exp is not None:
            got = d["block_energies_point_cm"][1] if len(
                d["block_energies_point_cm"]) > 1 else float("nan")
            print("    2P splitting {:.1f} cm^-1 (experiment {:.0f}) -- {}".format(
                got, exp, "valence shell" if 0.3 < got / exp < 3.0
                else "*** WRONG SHELL: check skip_pairs ***"), flush=True)
        print("    dE_ground={:+.3e} Eh  worst level shift {:.4f} cm^-1 ({:.2e} rel)  "
              "worst |dg| {:.3e} ({:.2e} rel)  [{:.0f}+{:.0f} s wall]".format(
                  d["delta_total_ground_eh"], d["worst_level_shift_cm"],
                  d["worst_level_shift_rel"], d["worst_g_shift_abs"],
                  d["worst_g_shift_rel"], point["wall_s"], gaussian["wall_s"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
