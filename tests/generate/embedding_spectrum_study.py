"""What an embedding does to a SPECTRUM, and at what lattice size.

**What this decides.** Whether Kuiva can be used for the crystal-embedded SMM work the
point-charge embedding exists for, and what field size that needs. The embedding is verified
as an *energy* (against PySCF's own QM/MM, to 2.8e-14 Eh), and the quantity it exists to
change is a *spectrum*: the ligand-field splittings and the g values of an ion in a lattice
against the same ion in vacuum. Nothing measured that, and nothing measured what a realistic
lattice (10^3-10^4 charges) costs end to end — the bare potential's batched grid integral had
never been run at that size.

Two phases, because they are two different questions:

1. ``spectrum`` — one 3d complex and one lanthanide complex, in vacuum and in a model lattice
   of increasing radius, compared through the phase-invariant reductions
   (:mod:`kuiva.props.multiplet`): level pattern in cm^-1 and the ground block's principal g
   values. The deliverable is the **convergence of the splittings with the field radius**.
2. ``cost`` — the ingestion cost against charge count out to 10^4, which is the size an
   Ewald-fitted lattice actually reaches. Nothing correlated runs: the embedding touches
   ``h1`` and a classical constant and nothing else, so the only term that can grow with the
   charge count is the one-electron build.

The model lattice
-----------------
⚠ **A model, stated as one, not a crystal structure.** A rock-salt arrangement of +-1 charges
on a cubic lattice of nearest-neighbour spacing ``LATTICE_D``, centred on the molecule, with every site inside
``EXCLUSION_R`` of any atom deleted so no charge sits in the molecular density (the embedding
refuses a charge on a nucleus, but a charge merely *near* one over-polarizes without
refusing). The field is then forced **neutral** by dropping the excess of whichever sign is in
surplus, outermost first: a non-neutral field is a monopole, and a monopole at the origin
shifts every level by an amount that has nothing to do with a crystal field.

Its purpose is a controlled, reproducible field with a *known* convergence behaviour — which
is what "at what lattice size" asks. A real structure is the separate generator entry this
study is the diagnostic for.

Run::

    python tests/generate/embedding_spectrum_study.py --phase cost
    python tests/generate/embedding_spectrum_study.py --phase spectrum --only ticl3

One system per invocation stays inside the ten-minute rule; records are written incrementally
after every radius.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import systems as sysdef                                                     # noqa: E402

from kuiva.interface import api                                              # noqa: E402
from kuiva.interface.environment import Environment                          # noqa: E402
from kuiva.props.dump import property_matrices                               # noqa: E402
from kuiva.util import resources as res                                      # noqa: E402
from progress import Heartbeat                                               # noqa: E402

EH_TO_CM = 219474.6313632
WALL_BUDGET_S = 9.5 * 60

#: ⚠ The rock-salt NEAREST-NEIGHBOUR spacing, Angstrom — not the conventional cubic cell
#: constant, which is twice it. Rock salt is a simple cubic array of alternating ions at
#: spacing d, so d is what sets the ion density (1/d^3); using the cell constant here makes a
#: lattice eight times too dilute and every field it produces eight times too weak. 2.8 A is
#: NaCl's own Na-Cl distance.
LATTICE_D = 2.8

#: No lattice site within this distance (Angstrom) of any atom of the molecule.
EXCLUSION_R = 4.0

#: Field radii swept, Angstrom. The last is ~2e3 charges; phase ``cost`` goes further.
RADII = (8.0, 12.0, 16.0, 20.0, 25.0)

#: (key, n_states). ticl3 is the 3d complex (5 Kramers doublets), cecl3 the lanthanide.
CASES = (("ticl3", 10), ("cecl3", 14))

#: name -> (builder, what the radius argument means). ``sphere`` is the naive cut; ``evjen``
#: is the cube with boundary weights, and the two together are the point of the comparison.
FIELD_MODELS: Dict[str, tuple] = {}

#: Selected by ``--model``; module-level so the phases need no extra parameter.
MODEL = "sphere"


def rocksalt_field(atoms, radius: float, *, a: float = LATTICE_D,
                   exclude: float = EXCLUSION_R) -> Tuple[np.ndarray, np.ndarray]:
    """A neutral rock-salt shell of +-1 charges between ``exclude`` and ``radius``.

    Coordinates in Angstrom, in the molecule's own frame, centred on the centroid of the
    atoms. Returns ``(charges, coords)`` — the array pair the embedding accepts directly,
    because a list of ten thousand tuples is not what a lattice generator should build.
    """
    centre = np.mean(np.asarray([xyz for _, xyz in atoms], dtype=float), axis=0)
    n = int(np.ceil(radius / a)) + 1
    grid = np.arange(-n, n + 1) * a
    x, y, z = np.meshgrid(grid, grid, grid, indexing="ij")
    pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    # rock salt: the sign alternates with the parity of the site index sum
    idx = np.rint(pts / a).astype(int)
    q = np.where((idx.sum(axis=1) % 2) == 0, 1.0, -1.0)
    pts = pts + centre
    r = np.linalg.norm(pts - centre, axis=1)
    keep = r <= radius
    # clear the molecule: no site within `exclude` of ANY atom
    for _, xyz in atoms:
        keep &= np.linalg.norm(pts - np.asarray(xyz, dtype=float), axis=1) > exclude
    q, pts, r = q[keep], pts[keep], r[keep]
    # force neutrality: drop the surplus sign from the outermost sites
    surplus = int(round(q.sum()))
    if surplus:
        sign = 1.0 if surplus > 0 else -1.0
        cand = np.where(q == sign)[0]
        drop = cand[np.argsort(-r[cand])][:abs(surplus)]
        mask = np.ones(q.size, dtype=bool)
        mask[drop] = False
        q, pts = q[mask], pts[mask]
    assert abs(q.sum()) < 1e-12, "the field must be neutral"
    return q, pts


def evjen_field(atoms, half_width: float, *, a: float = LATTICE_D,
                exclude: float = EXCLUSION_R) -> Tuple[np.ndarray, np.ndarray]:
    """The same rock-salt lattice cut as a CUBE with Evjen boundary weights.

    Evjen, *Phys. Rev.* **39**, 675 (1932): a site on a face of the bounding cube counts
    1/2, on an edge 1/4, at a corner 1/8 — i.e. the weight is ``(1/2)**k`` for a site lying on
    ``k`` of the three boundary planes. The cut is then made of neutral, dipole-free unit
    cells, and the Madelung potential converges in a handful of shells instead of
    conditionally.

    ⚠ **This is the reason a sphere cut is not enough.** Truncating an ionic lattice on a
    sphere leaves a surface whose multipole moments do not vanish with radius, so the potential
    at the centre oscillates rather than converges. A generator that reads a crystal structure
    and simply keeps every ion inside a cutoff inherits that, and the number it produces looks
    perfectly reasonable at every radius.
    """
    centre = np.mean(np.asarray([xyz for _, xyz in atoms], dtype=float), axis=0)
    n = int(np.rint(half_width / a))
    grid = np.arange(-n, n + 1) * a
    x, y, z = np.meshgrid(grid, grid, grid, indexing="ij")
    pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    idx = np.rint(pts / a).astype(int)
    q = np.where((idx.sum(axis=1) % 2) == 0, 1.0, -1.0)
    on_face = (np.abs(np.abs(idx) - n) == 0).sum(axis=1)      # 0..3 boundary planes
    q = q * (0.5 ** on_face)
    pts = pts + centre
    keep = np.ones(q.size, dtype=bool)
    for _, xyz in atoms:
        keep &= np.linalg.norm(pts - np.asarray(xyz, dtype=float), axis=1) > exclude
    q, pts = q[keep], pts[keep]
    # the exclusion breaks neutrality; restore it on the outermost retained sites, which
    # already carry the smallest weights
    net = float(q.sum())
    if abs(net) > 1e-12:
        r = np.linalg.norm(pts - centre, axis=1)
        order = np.argsort(-r)
        q[order[:64]] -= net / 64.0
    return q, pts


def _system(key: str) -> sysdef.System:
    for s in sysdef.SYSTEMS:
        if s.key == key:
            return s
    raise SystemExit("no system {!r}".format(key))


def _spectrum(system, n_states: int, field, *, memory_gb: float, label: str) -> Dict:
    """One full pipeline in the stated field; returns levels and phase-invariant reductions.

    ⚠ ``screening="none"`` throughout, on both sides. This is a *differential* measurement —
    vacuum against field — and the two-electron picture change is identical on both sides by
    construction, so paying a four-component atomic solve per element would buy nothing the
    difference can see. The one-electron SOC, which is what the g values need, stays on.
    """
    t0, c0 = time.time(), time.process_time()
    env = None if field is None else Environment(point_charges=field, unit="angstrom",
                                                 label=label)
    mol = api.Molecule(atoms=system.atoms, basis=system.basis, charge=system.charge,
                       spin=system.spin, environment=env)
    reference = api.spinor_reference(mol, screening="none", memory_gb=memory_gb)
    t_ingest = time.time() - t0
    outcome = api.casscf(reference, character=(system.atoms[0][0], system.active_l),
                         n_active=2 * system.ncas, n_active_elec=system.nelecas,
                         n_states=n_states, mode="second-order", conv_grad=1e-6,
                         report=False)
    props = reference.data.properties
    tdm = outcome.ci.transition_densities()
    matrices = property_matrices(outcome.coeff, outcome.active.spaces, tdm,
                                 outcome.ci.total_energies, props, reference.data.s_ao)
    energies = np.asarray(outcome.ci.total_energies, dtype=float)
    blocks = [{"size": int(b.size), "energy_cm": float(b.energy_cm),
               "spread_cm": float(b.spread_cm),
               "g_values": [float(g) for g in b.g_values]}
              for b in matrices.analyse(tol_cm=1.0)]
    return {
        "label": label,
        "n_charges": 0 if field is None else int(np.asarray(field[0]).size),
        "converged": bool(outcome.converged),
        "e_total": [float(e) for e in energies],
        "levels_cm": [float(x) for x in (energies - energies[0]) * EH_TO_CM],
        "blocks": blocks,
        "e_embedding_eh": float(getattr(reference.data, "e_embedding", 0.0) or 0.0),
        "ingest_wall_s": round(t_ingest, 2),
        "wall_s": round(time.time() - t0, 2), "cpu_s": round(time.process_time() - c0, 2),
    }


def _compare(vac: Dict, emb: Dict) -> Dict:
    """Field-induced shifts, through phase-invariant quantities only."""
    lv_v = np.asarray(vac["levels_cm"])
    lv_e = np.asarray(emb["levels_cm"])
    d = lv_e - lv_v
    g_shift = []
    for bv, be in zip(vac["blocks"], emb["blocks"]):
        for gv, ge in zip(bv["g_values"], be["g_values"]):
            g_shift.append(float(ge - gv))
    return {
        "n_charges": emb["n_charges"],
        "delta_levels_cm": [float(x) for x in d],
        "worst_level_shift_cm": float(np.max(np.abs(d))),
        "ground_block_g": [float(g) for g in emb["blocks"][0]["g_values"]],
        "vacuum_block_g": [float(g) for g in vac["blocks"][0]["g_values"]],
        "worst_g_shift": float(np.max(np.abs(g_shift))) if g_shift else 0.0,
        "block_energies_cm": [b["energy_cm"] for b in emb["blocks"]],
    }


def phase_spectrum(key: str, n_states: int, *, memory_gb: float, radii, out_path: str,
                   data: Dict, hb: Heartbeat) -> None:
    system = _system(key)
    rec = data["records"].setdefault("spectrum", {}).setdefault(
        key if MODEL == "sphere" else "{}:{}".format(key, MODEL), {})
    if "vacuum" not in rec:
        hb.tick(0, stage="{}: vacuum".format(key))
        res.clear()
        rec["vacuum"] = _spectrum(system, n_states, None, memory_gb=memory_gb,
                                  label="vacuum")
        print("    vacuum  levels {} cm^-1".format(
            " ".join("{:.1f}".format(x) for x in rec["vacuum"]["levels_cm"][:6])),
            flush=True)
        _save(out_path, data)
    for i, radius in enumerate(radii, start=1):
        tag = "R={:g}".format(radius)
        if tag in rec:
            continue
        if hb.expired:
            hb.tick(i, stage="wall budget exhausted before {} {}".format(key, tag))
            print("wall budget exhausted before {} {}".format(key, tag))
            return
        hb.tick(i, stage="{}: {}".format(key, tag))
        field = FIELD_MODELS[MODEL][0](system.atoms, radius)
        res.clear()
        rec[tag] = _spectrum(system, n_states, field, memory_gb=memory_gb,
                             label="{} d={:g} R={:g} A".format(MODEL, LATTICE_D, radius))
        rec[tag]["radius_a"] = float(radius)
        rec[tag]["delta"] = _compare(rec["vacuum"], rec[tag])
        d = rec[tag]["delta"]
        print("    {:8s} {:6d} charges  worst level shift {:9.2f} cm^-1  worst dg {:8.4f}  "
              "({:.0f} s wall, ingest {:.1f} s)".format(
                  tag, d["n_charges"], d["worst_level_shift_cm"], d["worst_g_shift"],
                  rec[tag]["wall_s"], rec[tag]["ingest_wall_s"]), flush=True)
        _save(out_path, data)


def phase_cost(key: str, *, memory_gb: float, out_path: str, data: Dict,
               hb: Heartbeat) -> None:
    """Ingestion cost against charge count. Nothing correlated runs: the embedding reaches
    ``h1`` and a classical constant, so this is the only term that grows with the count."""
    system = _system(key)
    rows = []
    for i, radius in enumerate((8.0, 12.0, 16.0, 20.0, 25.0, 30.0, 35.0, 40.0)):
        if hb.expired:
            hb.tick(i, stage="wall budget exhausted at R={}".format(radius))
            break
        field = FIELD_MODELS[MODEL][0](system.atoms, radius)
        n = int(field[0].size)
        hb.tick(i, stage="{}: cost at {} charges".format(key, n))
        res.clear()
        t0, c0 = time.time(), time.process_time()
        mol = api.Molecule(atoms=system.atoms, basis=system.basis, charge=system.charge,
                           spin=system.spin,
                           environment=Environment(point_charges=field, unit="angstrom"))
        data_scf = api.scalar_x2c_reference(mol, memory_gb=memory_gb, screening="none")
        rows.append({"key": key, "radius_a": float(radius), "n_charges": n,
                     "nao": int(data_scf.nao),
                     "e_embedding_eh": float(getattr(data_scf, "e_embedding", 0.0) or 0.0),
                     "e_scf": float(data_scf.e_scf),
                     "wall_s": round(time.time() - t0, 2),
                     "cpu_s": round(time.process_time() - c0, 2)})
        r = rows[-1]
        print("    R={:5.1f} A  {:6d} charges  ingest {:7.2f} s wall / {:7.2f} s cpu  "
              "E_qn={:+.6f} Eh".format(radius, n, r["wall_s"], r["cpu_s"],
                                       r["e_embedding_eh"]), flush=True)
        data["records"].setdefault("cost", {})[key] = rows
        _save(out_path, data)


def _save(path: str, data: Dict) -> None:
    with open(path + ".tmp", "w") as fh:
        json.dump(data, fh, indent=1, sort_keys=True)
    os.replace(path + ".tmp", path)


FIELD_MODELS.update(sphere=(rocksalt_field, "radius of the sphere, A"),
                    evjen=(evjen_field, "half-width of the cube, A"))


def main() -> int:
    global MODEL
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", required=True, choices=("spectrum", "cost"))
    ap.add_argument("--out", default="temp/embedding_spectrum.json")
    ap.add_argument("--only", default="", help="comma-separated system keys")
    ap.add_argument("--memory-gb", type=float, default=6.0)
    ap.add_argument("--radii", default="", help="comma-separated radii in Angstrom")
    ap.add_argument("--model", default="sphere", choices=("sphere", "evjen"),
                    help="sphere: naive cut. evjen: cube with boundary weights")
    args = ap.parse_args()
    MODEL = args.model
    only = {s for s in args.only.split(",") if s}
    radii = tuple(float(x) for x in args.radii.split(",")) if args.radii else RADII

    data: Dict[str, object] = {"lattice_d": LATTICE_D, "exclusion_r": EXCLUSION_R,
                               "radii": list(radii), "model": args.model, "records": {}}
    if os.path.exists(args.out):
        with open(args.out) as fh:
            data = json.load(fh)
    data.setdefault("records", {})
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    hb = Heartbeat("embedding_" + args.phase, budget_seconds=WALL_BUDGET_S,
                   meta={"phase": args.phase})
    for key, n_states in CASES:
        if only and key not in only:
            continue
        print("  {} [{}]".format(key, args.phase), flush=True)
        if args.phase == "spectrum":
            phase_spectrum(key, n_states, memory_gb=args.memory_gb, radii=radii,
                           out_path=args.out, data=data, hb=hb)
        else:
            phase_cost(key, memory_gb=args.memory_gb, out_path=args.out, data=data, hb=hb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
