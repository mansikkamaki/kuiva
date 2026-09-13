"""Can the selected cheap CI represent a coupled system's ground manifold at all?

What this answers
-----------------
The automatic selection now refuses a probe budget below the product of the sites' Hund
configurations (32 768 determinants for ``mn3_linear``'s three d^5 sites). That is a
necessary condition. Whether a selection *reaches* those determinants is a separate question:
the cheap CI grows its space by singles and doubles out of a few hundred leading determinants,
twice, from a reference CAS of at most 500 determinants over a window of the active space,
and the states of three coupled S = 5/2 ions are spread over spin configurations up to fifteen
spin flips away from any one of them.

This solves the cheap CI on the stored active-space integrals of ``mn3_linear`` (CAS(15, 30),
site-localized orbitals, no front end) at several budgets, with and without the Hund product
space as the seed, and reports each root's energy and ``<S^2>`` -- the Lieb-Mattis ground level
is S = 5/2 (``<S^2>`` = 8.75, six states), and a manifold whose spins come out non-integer or
whose multiplets are not degenerate is not a manifold.

Usage::

    source setup.sh
    python tests/generate/measure_cheap_ci_hund.py --out temp/cheap_ci_memory/hund.jsonl \\
        --budgets 6000,40000 --seeded-extra 8000
"""
import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from kuiva.ci.strings import Determinants, connections, rdm12              # noqa: E402
from kuiva.mcscf.preopt import cheap_ci                                     # noqa: E402
from kuiva.util.logging import set_verbosity                                # noqa: E402

INTEGRALS = ROOT / "temp" / "dmrg_campaign" / "mn3_linear_tier3_integrals.npz"


def hund_product_seed(sites, n_elec):
    """Every determinant with each Kramers pair of the (site-localized) shell singly occupied:
    one bit of each pair ``(2p, 2p+1)``. For a half-filled d^5 site that is its Hund space, and
    over the three sites the product space, ``2^15``."""
    n_pairs = len(sites) // 2
    assert n_elec == n_pairs, "this seed is the half-filled case only"
    masks = []
    for bits in product((0, 1), repeat=n_pairs):
        m = 0
        for p, b in enumerate(bits):
            m |= 1 << (2 * p + b)
        masks.append(m)
    return Determinants(masks=np.asarray(masks, dtype=np.uint64), n_spinor=2 * n_pairs,
                        n_elec=n_elec)


def spin_squared(ci, s_active, n):
    conn = connections(ci.dets)
    values = []
    for i in range(min(n, ci.civecs.shape[1])):
        gamma, gam2 = rdm12(ci.dets, ci.civecs[:, i], np.ones(1), conn)
        total = 0.0
        for s in s_active:
            total += np.sum((s @ s) * gamma) + np.einsum("pq,rs,pqrs->", s, s, gam2,
                                                         optimize=True)
        values.append(round(float(np.real(total)), 4))
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--budgets", default="6000,40000")
    parser.add_argument("--seeded-extra", type=int, default=0,
                        help="also solve seeded with the Hund product space plus this many")
    parser.add_argument("--n-states", type=int, default=16)
    parser.add_argument("--spin-roots", type=int, default=10,
                        help="<S^2> of the lowest this many roots (one 2-RDM each)")
    args = parser.parse_args()
    set_verbosity("WARNING")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    z = np.load(INTEGRALS)
    h, eri, n_elec, s_active = z["h"], z["eri"], int(z["n_elec"]), z["s_active"]

    def record(**row):
        with out.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print("#", json.dumps(row)[:400], flush=True)

    runs = [("unseeded", int(b), None) for b in args.budgets.split(",") if b]
    if args.seeded_extra:
        runs.append(("Hund-seeded", int(args.seeded_extra), hund_product_seed(z["sites"], n_elec)))
    for label, budget, seed in runs:
        c0, t0 = time.process_time(), time.time()
        ci = cheap_ci(h, eri, n_elec, n_states=args.n_states, max_determinants=budget,
                      with_2rdm=False, seed=seed)
        record(label=label, budget=budget, ndet=int(ci.n_determinants),
               relative_cm=[round(float(x), 2) for x in ci.relative_cm],
               s_squared=spin_squared(ci, s_active, args.spin_roots),
               cpu_s=round(time.process_time() - c0, 1), wall_s=round(time.time() - t0, 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
