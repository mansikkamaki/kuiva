"""Tier 3: derive and store the invariants the tensor-network tests assert against.

Unlike Tiers 1 and 2 this generator runs **no quantum chemistry and no external program**
(see :mod:`tier3_systems` for why no reference is possible). Everything written here is
either exact combinatorics on the system definitions, a theorem, or a dense exact
diagonalisation of a small effective spin model.

The file is still committed, for the same reason the other reference files are: it pins the
system definitions. Editing a spin, an oxidation state or an exchange pathway in
``tier3_systems.py`` changes these numbers, and the diff makes that visible instead of
silent.

Run:  python tests/generate/tier3_invariants.py
(no ``external/env.sh`` needed beyond NumPy). Writes ``tests/reference/tier3_invariants.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests/generate"))

import thermal                       # noqa: E402
import tier3_systems as t3           # noqa: E402

REF_OUT = REPO / "tests/reference/tier3_invariants.json"

SCHEMA = 2


def describe(system: t3.Tier3System) -> Dict:
    """Every invariant the tests use, derived from the definition alone."""
    parts = t3.bipartition(system)
    lm = t3.lieb_mattis_twice_spin(system)
    decomposition = t3.couple_spins([s.twice_spin for s in system.sites])

    rec: Dict = {
        "key": system.key,
        "label": system.label,
        "formula": system.formula,
        "topology": system.topology,
        "network_target": system.network_target,
        "provenance": system.provenance,
        "truncation": system.truncation,

        # --- sites -------------------------------------------------------------------
        "n_sites": system.n_sites,
        "sites": [{"label": s.label, "ion": s.ion, "twice_spin": s.twice_spin,
                   "kind": s.kind, "local_dim": s.local_dim} for s in system.sites],
        "local_dims": list(system.local_dims),
        "hilbert_dim": system.hilbert_dim,
        "total_unpaired_electrons": system.total_unpaired_electrons,
        "kramers_system": system.kramers_system,

        # --- graph -------------------------------------------------------------------
        "edges": [list(e) for e in system.edges],
        "n_edges": len(system.edges),
        "degree_sequence": list(t3.degree_sequence(system)),
        "connected_components": t3.connected_components(system),
        "cycle_rank": t3.cycle_rank(system),
        "is_tree": t3.is_tree(system),
        "diameter": t3.graph_diameter(system),
        "bipartite": parts is not None,
        "sublattices": [list(parts[0]), list(parts[1])] if parts else None,
        "frustrated": t3.is_frustrated(system),

        # --- exact spin algebra ------------------------------------------------------
        "spin_decomposition": {str(k): v for k, v in decomposition.items()},
        "lieb_mattis_twice_spin": lm,
        "experimental_twice_spin": system.experimental_twice_spin,
        "isotropic_exchange": system.isotropic_exchange,

        # --- why there is no reference calculation ------------------------------------
        "cas_electrons": system.cas_electrons,
        "cas_orbitals": system.cas_orbitals,
        "cas_spinors": system.cas_spinors,
        "cas_determinants": system.cas_determinants,
        "beyond_conventional_ci": system.beyond_conventional_ci,
    }

    ed = t3.heisenberg_ground_state(system)
    rec["heisenberg_ed"] = ed
    return rec


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="", help="comma-separated system keys")
    ap.add_argument("--merge", action="store_true",
                    help="merge into the existing reference file instead of replacing it")
    args = ap.parse_args(argv)

    out: Dict = {
        "schema": SCHEMA,
        "generator": "tests/generate/tier3_invariants.py",
        "note": "Derived invariants only - no external program can produce a reference for "
                "these systems (see tier3_systems.py). Values are exact combinatorics, "
                "theorems, or dense ED of an effective spin model.",
        "conventional_ci_spinor_ceiling": t3.CONVENTIONAL_CI_SPINOR_CEILING,
        "ed_max_dim": t3.ED_MAX_DIM,
        "environment": thermal.describe_environment(),
        "records": {},
    }
    if args.merge and REF_OUT.is_file():
        out = json.loads(REF_OUT.read_text())
        out.setdefault("records", {})

    keys = [k for k in args.only.split(",") if k] or None
    for system in t3.SYSTEMS:
        if keys and system.key not in keys:
            continue
        with thermal.track_resources() as res:
            rec = describe(system)
        rec["resources"] = res.as_dict()
        out["records"][system.key] = rec

        ed = rec["heisenberg_ed"]
        ed_txt = ("ED 2S=%d deg=%d" % (ed["twice_total_spin"], ed["degeneracy"])
                  if ed else "ED skipped")
        lm_txt = ("LM 2S=%s" % rec["lieb_mattis_twice_spin"]
                  if rec["lieb_mattis_twice_spin"] is not None else "LM n/a (frustrated/SOC)")
        print(f"[tier3] {system.key:14s} {system.topology:28s} "
              f"dim={rec['hilbert_dim']:6d} cyc={rec['cycle_rank']} "
              f"{lm_txt:26s} {ed_txt:20s} CAS dets={rec['cas_determinants']:.3e}",
              flush=True)

    REF_OUT.parent.mkdir(parents=True, exist_ok=True)
    REF_OUT.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"\nwrote {REF_OUT.relative_to(REPO)}  ({len(out['records'])} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
