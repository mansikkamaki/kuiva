"""Measure the probe's budget-to-budget noise floor on the target spectrum.

What this answers
-----------------
``kuiva/autocas/protocol.py`` decides whether a feature class stays by comparing two probes'
*target spectra* -- the relative energies inside the theoretical ground manifold, plus the gap
to the next one. That comparison is only worth making if the probe's own uncertainty is
smaller than the splittings it has to resolve, and the two constants that encode the answer
(``DEFAULT_SPECTRUM_TOL_CM``, ``DEFAULT_PROBE_NOISE_CM``) may not be guesses.

What is varied is the **budget**, because that is the probe's only real freedom: the
determinant count its selected CI may hold and the macro-iterations it may spend. Two probes
of the same space at different budgets differ by exactly the amount a probe's spectrum is
uncertain by -- and the round loop compares probes at the *same* budget, so this is an upper
bound on what it sees.

⚠ **Not a convergence study.** The probe is a truncated CI in a truncated space and its
energies are qualitative by construction; nothing here says a larger budget is closer to
anything. What it says is how far the number the protocol reads moves when the one knob
behind it moves.

Usage (each invocation is one system, so each stays inside the ten-minute rule)::

    source setup.sh
    python tests/generate/measure_autocas_noise.py ticl3 --out notes/ticl3.json

Results are written **incrementally**, one probe per line, so a run that is killed still
leaves everything it had measured.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "generate"))

import systems as sysdef                                                    # noqa: E402

from kuiva.autocas import candidates as cand                                # noqa: E402
from kuiva.autocas import centres as ctr                                    # noqa: E402
from kuiva.autocas import protocol as proto                                 # noqa: E402
from kuiva.autocas.probe import ProbeBudget, probe, probe_roots             # noqa: E402
from kuiva.autocas.roots import target_spectrum                             # noqa: E402
from kuiva.interface import api                                             # noqa: E402
from kuiva.util.logging import set_verbosity                                # noqa: E402

#: Probes run, in order. The first three are the **same** budget three times -- the default
#: the protocol ships with -- because two probes of one space at one budget need not agree:
#: the pre-optimizer stops on its iteration budget rather than at a stationary point, and a
#: threaded reduction's order is not fixed, so the trajectory can differ in its last bits and
#: a non-stationary point amplifies that. ⚠ **That repeat spread is the real noise floor**;
#: the budget spread below it is the (larger) uncertainty of the probe as an instrument.
#:
#: The determinant budget is on the list once and is a no-op wherever the active space is
#: small enough to be complete at any of these values; what moves the answer is the iteration
#: count.
BUDGETS = [
    dict(max_iter=8, max_determinants=6000),
    dict(max_iter=8, max_determinants=6000),
    dict(max_iter=8, max_determinants=6000),
    dict(max_iter=4, max_determinants=6000),
    dict(max_iter=12, max_determinants=6000),
    dict(max_iter=16, max_determinants=6000),
    dict(max_iter=8, max_determinants=2000),
]


def control_pairs(reference, space, *, n_deep: int = 1, n_high: int = 1):
    """Kramers pairs that cannot describe anything the ground manifold is made of.

    The **control** of the whole measurement. Two probes of one space at one budget agree
    bitwise, so the number the round loop actually reads -- the distance between a probe of
    the core and a probe of the core *plus a class* -- has a noise component that repeats
    cannot expose: adding orbitals changes the determinant selection and the optimizer's
    trajectory, and the pre-optimizer stops on its iteration budget rather than at a
    stationary point.

    So this adds orbitals whose *physics* is known to be irrelevant -- the deepest inactive
    pairs (a chlorine 1s) and the highest virtual ones -- and measures what the target
    spectrum does anyway. ⚠ That is an **upper bound on the noise and not the noise itself**:
    core-valence correlation is a real effect, merely a tiny one at this level of
    correlation. A keep/drop threshold below this number would be reading the trajectory.
    """
    inactive = np.asarray(space.spaces.inactive, dtype=int)
    virtual = np.asarray(space.spaces.virtual, dtype=int)
    deep = inactive[:2 * int(n_deep)]
    high = virtual[-2 * int(n_high):] if n_high else np.zeros(0, dtype=int)
    return np.asarray(deep, dtype=int), np.asarray(high, dtype=int)


def control(args, reference, construction, space, floor_target, n_roots, out, started):
    """Probe the core, then the core plus an irrelevant pair, and report the distance."""
    from kuiva.mcscf.casci import active_space

    budget = ProbeBudget(**BUDGETS[0])
    base = probe(reference, construction.coeff, space, n_roots=n_roots, budget=budget)
    base_target = target_spectrum(base.spectrum_cm, floor_target)
    rows = [{"added": "none", "cas": [space.n_elec, space.n_active],
             "target_cm": [round(float(x), 3) for x in base_target],
             "cpu_s": round(base.cpu_seconds, 1)}]
    n = int(args.control_pairs)
    deep, high = control_pairs(reference, space, n_high=n)
    virtual = np.asarray(space.spaces.virtual, dtype=int)
    if construction.double is not None:
        # The double shell's own pairs sit at the top of the empty group's projection order,
        # i.e. right after the shell: a control that drew them would be the class itself.
        virtual = np.setdiff1d(virtual, np.asarray(construction.double.columns, dtype=int))
    low = virtual[:2 * n]
    high = virtual[-2 * n:]
    trials = [("deepest inactive pair", deep, 2), ("highest virtual pair", high, 0)]
    if n > 1:
        # ⚠ The control a correlating (double) shell is judged against: the same NUMBER of
        # added pairs, drawn from the empty orbitals nearest the gap and from the top of the
        # virtual space, neither of which is the shell's radial partner.
        trials = [("{} lowest virtual pairs".format(n), low, 0),
                  ("{} highest virtual pairs".format(n), high, 0)]
    for label, extra, electrons in trials:
        if extra.size == 0 or time.time() - started > args.wall:
            continue
        columns = np.sort(np.concatenate([np.asarray(space.spaces.active, dtype=int), extra]))
        grown = active_space(columns, int(reference.nspinor),
                             int(reference.data.nelec_total),
                             n_active_elec=int(space.n_elec + electrons))
        result = probe(reference, construction.coeff, grown, n_roots=n_roots,
                       budget=budget)
        target = target_spectrum(result.spectrum_cm, floor_target)
        distance = (float("inf") if target.size != base_target.size
                    else float(np.max(np.abs(target - base_target))))
        rows.append({"added": label, "cas": [grown.n_elec, grown.n_active],
                     "target_cm": [round(float(x), 3) for x in target],
                     "distance_cm": None if not np.isfinite(distance) else round(distance, 3),
                     "cpu_s": round(result.cpu_seconds, 1)})
        print("# control: + {} -> d = {} cm^-1".format(label, rows[-1]["distance_cm"]),
              flush=True)
    with out.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row, kind="control")) + "\n")
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("system")
    parser.add_argument("--out", default=None)
    parser.add_argument("--budgets", type=int, default=len(BUDGETS),
                        help="how many of the budget list to run (a wall-time bound)")
    parser.add_argument("--wall", type=float, default=600.0,
                        help="hard wall budget [s]; the run stops between probes")
    parser.add_argument("--control", action="store_true",
                        help="run the control instead: the core, then the core plus a "
                             "physically irrelevant pair")
    parser.add_argument("--control-pairs", type=int, default=1,
                        help="with --control: how many virtual pairs to add (a double "
                             "shell's control adds as many as the shell has)")
    args = parser.parse_args(argv)
    set_verbosity("WARNING")

    out = Path(args.out) if args.out else ROOT / "temp" / (
        "autocas_noise_{}.jsonl".format(args.system))
    out.parent.mkdir(parents=True, exist_ok=True)

    system = sysdef.get(args.system)
    started = time.time()
    molecule = api.Molecule(atoms=system.atoms, basis=system.basis, charge=system.charge,
                            spin=system.spin)
    data = api.scalar_x2c_reference(molecule, screening="none", memory_gb=8.0,
                                    atomic_reference=True)
    reference = api.spinor_reference(data, memory_gb=8.0)
    front_end = time.time() - started
    print("# {}: front end {:.1f} s wall".format(args.system, front_end), flush=True)

    centres = ctr.detect_centres(reference, report=False).centres
    # With a multi-pair control the construction is the one a double-shell round uses, so the
    # control and the class are probed from the same rotated orbitals.
    construction = cand.shell_candidates(reference, centres, report=False,
                                         double=bool(args.control and args.control_pairs > 1))
    shell = construction.shell
    space = proto._space_of(reference, [shell])
    floor, product, terms = proto._floors(centres, shell)
    floor_target = max(1, min(product, 64))
    n_roots = probe_roots(floor_target, n_elec=space.n_elec, n_active=space.n_active)
    print("# CAS({}, {}), floor {} (product {}, {}), {} roots"
          .format(space.n_elec, space.n_active, floor, product, terms, n_roots), flush=True)

    if args.control:
        out.write_text(json.dumps({"system": args.system, "kind": "control",
                                   "cas": [space.n_elec, space.n_active],
                                   "budget": BUDGETS[0]}) + "\n")
        control(args, reference, construction, space, floor_target, n_roots, out, started)
        print("# total {:.1f} s wall".format(time.time() - started), flush=True)
        return 0

    rows = []
    with out.open("w") as handle:
        handle.write(json.dumps({"system": args.system, "cas": [space.n_elec,
                                                                space.n_active],
                                 "floor": floor, "product": product, "terms": terms,
                                 "n_roots": n_roots,
                                 "front_end_wall_s": round(front_end, 1)}) + "\n")
        handle.flush()
        for budget_options in BUDGETS[:args.budgets]:
            if time.time() - started > args.wall:
                print("# wall budget reached; stopping between probes", flush=True)
                break
            budget = ProbeBudget(**budget_options)
            result = probe(reference, construction.coeff, space, n_roots=n_roots,
                           budget=budget)
            target = target_spectrum(result.spectrum_cm, floor_target)
            row = {"budget": budget_options,
                   "cpu_s": round(result.cpu_seconds, 1),
                   "wall_s": round(result.wall_seconds, 1),
                   "ndet": result.n_determinants,
                   "converged": bool(result.converged),
                   "target_cm": [round(float(x), 3) for x in target],
                   "spectrum_cm": [round(float(x), 3) for x in result.spectrum_cm]}
            rows.append(row)
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            print("# {} -> target {} ({:.1f} s cpu)".format(budget_options,
                                                            row["target_cm"], row["cpu_s"]),
                  flush=True)

        spread = None
        if len(rows) > 1:
            base = np.asarray(rows[0]["target_cm"], dtype=float)

            def distance(row):
                other = np.asarray(row["target_cm"], dtype=float)
                return (float("inf") if other.size != base.size
                        else float(np.max(np.abs(other - base))))

            distances = [distance(row) for row in rows[1:]]
            spread = distances
            repeats = [d for d, row in zip(distances, rows[1:])
                       if row["budget"] == rows[0]["budget"]]
            handle.write(json.dumps({
                "distances_from_first_cm": [None if not np.isfinite(d) else round(d, 3)
                                            for d in distances],
                "repeat_spread_cm": (None if not repeats or not np.isfinite(max(repeats))
                                     else round(float(max(repeats)), 3)),
                "budget_spread_cm": (None if len(distances) <= len(repeats)
                                     or not np.isfinite(max(distances[len(repeats):]))
                                     else round(float(max(distances[len(repeats):])), 3)),
            }) + "\n")
            handle.flush()
    print("# distances from the first probe [cm^-1]: {}".format(spread), flush=True)
    print("# total {:.1f} s wall".format(time.time() - started), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
