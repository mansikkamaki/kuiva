"""Does a keep/drop verdict taken at FIXED orbitals remove the size-growing probe noise?

What this answers
-----------------
The round loop compares a probe of the accepted space with a probe of the space plus one
class. Both probes pre-optimize their orbitals, so a class of four or seven pairs nobody would
call relevant still moves the target spectrum by 58-102 cm^-1: a pre-optimization stopped on
its iteration budget rotates into whatever it is given. Two ways out were named -- a
reproducible control of the class's size, or a verdict at orbitals the addition cannot
change -- and this script measures the second, in two variants:

``construction``
    Both CIs at the AVAS-rotated SCF orbitals the candidates were built in. No optimization
    at all; the cheapest and the most reproducible.
``probe``
    The core is probed as today (orbitals optimized for the accepted space), and the trial
    space is the core's probe orbitals with the class's orbitals **carried** into them:
    projected onto the probe's inactive (occupied pairs) or virtual (empty pairs) span,
    rebuilt as Kramers pairs, the complement kept. Both CIs at those orbitals.

For each variant: the class itself, and controls of 1, 2, 4 and 7 pairs -- the deepest
inactive, the highest virtual, and the lowest virtual that is not the class. ``--trajectory``
also probes core + class the way the protocol does today, for comparison in one process.

Usage (one system per invocation, each inside the ten-minute rule)::

    source setup.sh
    python tests/generate/measure_autocas_fixed.py ticl3 --out temp/autocas_fixed/ticl3.jsonl
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
from kuiva.autocas import targets as tg                                     # noqa: E402
from kuiva.autocas.probe import ProbeBudget, probe, probe_roots             # noqa: E402
from kuiva.autocas.roots import target_spectrum                             # noqa: E402
from kuiva.interface import api                                             # noqa: E402
from kuiva.mcscf.casci import active_space                                  # noqa: E402
from kuiva.mcscf.orbopt import CASIntegrals, OrbitalSpaces                  # noqa: E402
from kuiva.mcscf.preopt import cheap_ci                                     # noqa: E402
from kuiva.spinor.expand import nearest_kramers_paired, spin_block_diagonal  # noqa: E402
from kuiva.util.logging import set_verbosity                                # noqa: E402


def fixed_spectrum(reference, coeff, spaces, n_elec, n_roots, max_det=6000):
    ints = CASIntegrals.build(reference.factors, reference.h_one_electron(), coeff, spaces,
                              e_nuc=reference.data.e_nuc)
    ci = cheap_ci(ints.h_active_effective(), ints.active_eri(), int(n_elec),
                  n_states=int(n_roots), max_determinants=max_det, with_2rdm=False)
    return np.asarray(ci.relative_cm, dtype=float), int(ci.n_determinants)


def carry(reference, base_coeff, base_spaces, extra_ao, occupied_pairs):
    """The base orbitals with ``extra_ao``'s pairs carried into the active space.

    Returns ``(coeff, spaces, retained)``: ``retained`` is the smallest singular value of the
    extra orbitals' projection onto the span they were carried into (1 = nothing lost).
    """
    x2 = spin_block_diagonal(np.asarray(reference.orth.x))
    s2 = spin_block_diagonal(np.asarray(reference.data.s_ao))
    to_work = lambda c: x2.conj().T @ s2 @ np.asarray(c)            # noqa: E731
    work = to_work(base_coeff)
    extra = to_work(extra_ao)
    ina, act, vir = (np.asarray(base_spaces.inactive, dtype=int),
                     np.asarray(base_spaces.active, dtype=int),
                     np.asarray(base_spaces.virtual, dtype=int))
    occ_cols = np.asarray([c for p in occupied_pairs for c in (2 * p, 2 * p + 1)], dtype=int)
    emp_cols = np.setdiff1d(np.arange(extra.shape[1]), occ_cols)
    blocks_in = {"I": work[:, ina], "V": work[:, vir]}
    new_parts, retained = [], 1.0
    for key, cols in (("I", occ_cols), ("V", emp_cols)):
        if cols.size == 0:
            continue
        target = blocks_in[key]
        u, s, _ = np.linalg.svd(target.conj().T @ extra[:, cols], full_matrices=True)
        k = cols.size
        retained = min(retained, float(s[-1]))
        new_parts.append(target @ u[:, :k])
        blocks_in[key] = target @ u[:, k:]
    added = np.concatenate(new_parts, axis=1)
    c = np.concatenate([blocks_in["I"], work[:, act], added, blocks_in["V"]], axis=1)
    n_i, n_a, n_v = blocks_in["I"].shape[1], act.size + added.shape[1], blocks_in["V"].shape[1]
    idx_i = np.arange(n_i)
    idx_a0 = np.arange(n_i, n_i + act.size)
    idx_a1 = np.arange(n_i + act.size, n_i + n_a)
    idx_v = np.arange(n_i + n_a, n_i + n_a + n_v)
    c = nearest_kramers_paired(c, (idx_a0, idx_a1, idx_i, idx_v))
    spaces = OrbitalSpaces(inactive=idx_i, active=np.arange(n_i, n_i + n_a),
                           virtual=idx_v, n_orb=c.shape[1])
    return np.ascontiguousarray(x2 @ c), spaces, retained


def canonical(reference, coeff, spaces, exclude=()):
    """Rotate the inactive and virtual pairs (minus ``exclude``) to eigenvectors of the
    Kramers-pair-folded inactive Fock at these orbitals; return ``(coeff, inactive_by_energy,
    virtual_by_energy)`` as spinor-column arrays. A basis that is unique wherever the orbital
    energies are not degenerate -- which a control must be drawn from."""
    from kuiva.spinor.expand import fold_to_kramers_pairs, rotate_kramers_pairs
    ints = CASIntegrals.build(reference.factors, reference.h_one_electron(), coeff, spaces,
                              e_nuc=reference.data.e_nuc)
    banned = set(int(c) for c in np.asarray(exclude, dtype=int))
    c = np.array(coeff, copy=True)
    orders = []
    for block in (spaces.inactive, spaces.virtual):
        cols = np.asarray([x for x in np.asarray(block, dtype=int) if x not in banned])
        f_pair, _ = fold_to_kramers_pairs(ints.f_inactive, columns=cols)
        w, v = np.linalg.eigh(f_pair)
        c = rotate_kramers_pairs(c, v, cols)
        orders.append(cols)                      # columns now hold ascending-energy vectors
    return c, orders[0], orders[1]


def distance(a, b):
    return float("inf") if a.size != b.size else float(np.max(np.abs(a - b)))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("system")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-iter", type=int, default=8)
    parser.add_argument("--sizes", default="1,2,4,7")
    parser.add_argument("--trajectory", action="store_true")
    parser.add_argument("--wall", type=float, default=560.0)
    parser.add_argument("--variants", default="AB")
    parser.add_argument("--nested", action="store_true",
                        help="variant A seeds every trial with the core's determinants")
    parser.add_argument("--max-determinants", type=int, default=6000)
    args = parser.parse_args(argv)
    set_verbosity("WARNING")
    started = time.time()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("")

    def record(**row):
        row["elapsed_s"] = round(time.time() - started, 1)
        with out.open("a") as fh:
            fh.write(json.dumps(row, default=float) + "\n")
        print("#", json.dumps(row, default=float)[:300], flush=True)

    system = sysdef.get(args.system)
    molecule = api.Molecule(atoms=system.atoms, basis=system.basis, charge=system.charge,
                            spin=system.spin)
    data = api.scalar_x2c_reference(molecule, screening="none", memory_gb=8.0,
                                    atomic_reference=True)
    reference = api.spinor_reference(data, memory_gb=8.0)
    centres = ctr.detect_centres(reference, report=False).centres
    bridge = args.system == "ti2cl6"
    construction = cand.shell_candidates(reference, centres, report=False, double=not bridge,
                                         bonding=1 if args.system == "fecl2" else 0)
    coeff0 = construction.coeff
    shell = construction.shell
    classes = []
    if construction.double is not None and not construction.double.empty:
        classes.append(construction.double)
    if construction.bonding is not None and not construction.bonding.empty:
        classes.append(construction.bonding)
    if bridge:
        projection = proto._Projection(
            eigenvalues=np.asarray(construction.avas.eigenvalues, float),
            occupations=np.asarray(construction.avas.occupations, float), coeff=coeff0,
            selected=np.asarray(construction.avas.selected, dtype=int))
        target = [t for t in tg.parse_targets(["shells", ("bridge", ("Ti1", "Ti2"))])
                  if t.cls == "bridge"][0]
        classes.append(cand.bridge_candidates(reference, target, projection,
                                              exclude=list(shell.columns), report=False))
    space = proto._space_of(reference, [shell])
    floor, product, terms = proto._floors(centres, shell)
    floor_target = max(1, min(product, 64))
    n_roots = probe_roots(floor_target, n_elec=space.n_elec, n_active=space.n_active)
    occ = np.asarray(reference.spinors.occ, dtype=float)
    record(event="setup", system=args.system, cas=[space.n_elec, space.n_active],
           floor=floor_target, n_roots=n_roots, terms=terms,
           classes=[(c.cls, c.n_pairs) for c in classes])

    class_cols = np.concatenate([np.asarray(c.columns, dtype=int) for c in classes]) \
        if classes else np.zeros(0, dtype=int)
    coeff0, ina, vir_free = canonical(reference, coeff0, space.spaces, exclude=class_cols)
    sizes = [int(x) for x in args.sizes.split(",")]

    def control_trials(ina, vir):
        rows = []
        for n in sizes:
            rows.append(("deep inactive x{}".format(n), ina[:2 * n]))
            rows.append(("high virtual x{}".format(n), vir[-2 * n:]))
            rows.append(("low virtual x{}".format(n), vir[:2 * n]))
        return rows

    trials = [(c.cls, np.asarray(c.columns, dtype=int)) for c in classes] \
        + control_trials(ina, vir_free)

    def grown_space(cols):
        columns = np.sort(np.concatenate([np.asarray(space.spaces.active, dtype=int), cols]))
        electrons = space.n_elec + int(round(float(np.sum(occ[cols]))))
        return active_space(columns, int(reference.nspinor), int(reference.data.nelec_total),
                            n_active_elec=electrons)

    # --- variant A: the construction orbitals ------------------------------------------
    from kuiva.autocas.probe import measure
    budget_a = ProbeBudget(max_determinants=args.max_determinants)
    t = time.process_time()
    core_a = measure(reference, coeff0, space, n_roots=n_roots, budget=budget_a)
    base_a, ndet = core_a.spectrum_cm, core_a.n_determinants
    ref_a = target_spectrum(base_a, floor_target)
    record(event="A core", target=ref_a.tolist(), ndet=ndet,
           cpu_s=round(time.process_time() - t, 1))
    for label, cols in (trials if "A" in args.variants else []):
        if time.time() - started > args.wall:
            record(event="stopped", reason="wall")
            return 0
        t = time.process_time()
        g = grown_space(cols)
        m = measure(reference, coeff0, g, n_roots=n_roots, budget=budget_a,
                    seed=core_a if args.nested else None)
        spec, ndet = m.spectrum_cm, m.n_determinants
        tgt = target_spectrum(spec, floor_target)
        record(event="A trial", nested=bool(args.nested), label=label, cas=[g.n_elec, g.n_active],
               d_cm=distance(ref_a, tgt), target=tgt.tolist(), ndet=ndet,
               cpu_s=round(time.process_time() - t, 1))

    # --- variant B: the core probe's orbitals, the trial carried into them -------------
    budget = ProbeBudget(max_iter=args.max_iter)
    if "B" not in args.variants:
        record(event="done")
        return 0
    core = probe(reference, coeff0, space, n_roots=n_roots, budget=budget)
    record(event="B core probe", cpu_s=round(core.cpu_seconds, 1),
           wall_s=round(core.wall_seconds, 1),
           probe_target=target_spectrum(core.spectrum_cm, floor_target).tolist())
    probe_coeff, ina_b, vir_b = canonical(reference, core.coeff, space.spaces)
    base_b, ndet = fixed_spectrum(reference, probe_coeff, space.spaces, space.n_elec, n_roots)
    ref_b = target_spectrum(base_b, floor_target)
    record(event="B core", target=ref_b.tolist(), ndet=ndet)
    trials_b = [(c.cls, np.asarray(c.columns, dtype=int), coeff0) for c in classes] \
        + [(label, cols, probe_coeff) for label, cols in control_trials(ina_b, vir_b)]
    for label, cols, source in trials_b:
        if time.time() - started > args.wall:
            record(event="stopped", reason="wall")
            return 0
        t = time.process_time()
        pairs = cols[0::2] // 2
        if source is coeff0:
            occupied = [i for i, p in enumerate(pairs) if occ[2 * p] > 0.5]
        else:
            occupied = [i for i, p in enumerate(pairs) if 2 * p in set(ina_b.tolist())]
        c_b, sp_b, kept = carry(reference, probe_coeff, space.spaces, source[:, cols], occupied)
        n_elec = space.n_elec + 2 * len(occupied)
        spec, ndet = fixed_spectrum(reference, c_b, sp_b, n_elec, n_roots)
        tgt = target_spectrum(spec, floor_target)
        record(event="B trial", label=label, cas=[n_elec, sp_b.active.size],
               d_cm=distance(ref_b, tgt), retained=round(kept, 4), target=tgt.tolist(),
               ndet=ndet, cpu_s=round(time.process_time() - t, 1))

    if args.trajectory:
        ref_t = target_spectrum(core.spectrum_cm, floor_target)
        for c in classes:
            g = grown_space(np.asarray(c.columns, dtype=int))
            trial = probe(reference, coeff0, g, n_roots=n_roots, budget=budget)
            tgt = target_spectrum(trial.spectrum_cm, floor_target)
            record(event="trajectory trial", label=c.cls, d_cm=distance(ref_t, tgt),
                   target=tgt.tolist(), cpu_s=round(trial.cpu_seconds, 1))
    record(event="done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
