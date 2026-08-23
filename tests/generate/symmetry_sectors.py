"""S4/S6 measurements for abelian double-group symmetry: what sector restriction is worth.

Two questions, both answered on CPU seconds and both bounded by a hard wall budget so the run
terminates whether or not it converges — an exploratory run is designed to finish inside ten
minutes, and a case list is what keeps it there.

**S4 — is a sector-restricted determinant space worth building?**  Three candidate designs
were on the table: (a) a per-sector ``Determinants`` list with its own ``O(N^2)``
connection scan, (b) a full-space solve with per-sector guesses and classification only, and
(c) a closed-form sector rank table. What shipped is between (a) and (b): the **eigenproblem**
is compressed to the sector while the **sigma vector** stays a full-space contraction, so a
request for ``n`` roots of a sector costs ``n`` roots rather than however many roots of the
full spectrum lie below them, at a full-space cost per application of ``H``. This measures
that against the general path at matched physics — the same states, reached two ways — which
is what says whether (a)'s extra machinery has anything left to win.

**S6 — does per-irrep selection make the sub-manifold-average question moot?**  It was
recorded as a hope, and this is the measurement. For a symmetric system it reports, per
selection, the state-average boundary gap and the spin non-invariance of the averaged
density, so "the selection can no longer cut a manifold" and "the ensemble is one the
symmetry leaves invariant" are stated separately — they are different claims and only the
second is what a sub-manifold average lacks.

Run::

    source setup.sh && python tests/generate/symmetry_sectors.py [--budget 540]

Results are appended incrementally (never only at the end) as JSON lines, and are tabulated
in the CI and MCSCF packages' validation notes.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kuiva.interface.api import Molecule, active_space_for, spinor_reference
from kuiva.mcscf.casci import FullCISolver, state_average_boundary
from kuiva.mcscf.orbopt import CASIntegrals
from kuiva.spinor.expand import spin_operator
from kuiva.symm.sectors import sector_violation
from kuiva.util.logging import set_verbosity

import systems


def _cpu() -> float:
    return time.process_time()


def _reference(system, point_group="auto", memory_gb=8.0):
    mol = Molecule(atoms=list(system.atoms), basis=system.basis, charge=system.charge,
                   spin=system.spin, point_group=point_group)
    return spinor_reference(mol, screening="none", memory_gb=memory_gb)


def _integrals(ref, space):
    return CASIntegrals.build(ref.factors, ref.h_one_electron(), ref.spinors_in_ao(),
                              space.spaces, e_nuc=ref.data.e_nuc)


def _time_solve(make_solver, ints, repeats=2):
    """CPU seconds and applications of H for the cheapest of ``repeats`` solves.

    The cheapest, not the mean: what is being compared is the work the algorithm does, and a
    slower repeat on this box is a statement about the thermal envelope rather than about the
    method (wall time here is not a cost measure at all).
    """
    best = None
    for _ in range(repeats):
        solver = make_solver()
        t0 = _cpu()
        solver.solve(ints)
        cpu = _cpu() - t0
        record = (cpu, solver.last.n_apply, float(solver.last.energies[0]))
        best = record if best is None or record[0] < best[0] else best
    return best


#: Active spaces for S4, chosen to straddle the Davidson/dense boundary: the ligand-field
#: case is 10 determinants and is solved densely (where the comparison is vacuous by
#: construction and is recorded as such), the N2 spaces are genuine iterative solves.
S4_CASES = (("ticl3", ("Ti", "d"), 10, None),
            ("n2", None, 12, 8),
            ("n2", None, 14, 8),
            ("n2", None, 16, 8))


def _n2_reference():
    from kuiva.interface.api import Molecule
    mol = Molecule(atoms=[("N", (0.0, 0.0, 0.55)), ("N", (0.0, 0.0, -0.55))],
                   basis="x2c-SVPall-2c", point_group="auto")
    return spinor_reference(mol, screening="none", memory_gb=8.0)


def measure_s4(out, budget_end):
    """Per-irrep selection against the general path, at matched physics.

    ⚠ Matched physics, not matched root counts: a general request for ``n`` roots and a
    per-irrep request summing to ``n`` are the same amount of *answer* only when the sectors
    are the ones the general spectrum's lowest ``n`` fall into, which is why the request is
    built from the general run's own classification.
    """
    cases = []
    references = {}
    for key, character, n_active, n_elec in S4_CASES:
        # ⚠ The budget is checked before starting a case, not inside one: a case that has
        # begun runs to the end, so the *design* of the case list is what keeps the run
        # inside ten minutes. The wrapper timeout is the backstop, not the plan.
        if time.time() > budget_end:
            out({"note": "S4 stopped on the wall budget", "done": len(cases)})
            break
        if key not in references:
            references[key] = (_n2_reference() if key == "n2"
                               else _reference(systems.get(key)))
        ref = references[key]
        if character is not None:
            space = active_space_for(ref, character=character, n_active=n_active)
        else:
            first = ref.data.nelec_total - n_elec
            space = active_space_for(ref, active=list(range(first, first + n_active)),
                                     n_active_elec=n_elec)
        ints = _integrals(ref, space)
        group = space.labels.group
        h = np.ascontiguousarray(ints.h_active_effective())
        eri = ints.active_eri()
        n_states = 6
        general = _time_solve(
            lambda: FullCISolver(space.spaces.n_active, space.n_elec, n_states=n_states,
                                 symmetry=space.labels, enforce_kramers=False), ints)
        # the sectors the general spectrum's lowest n_states actually land in
        probe = FullCISolver(space.spaces.n_active, space.n_elec, n_states=n_states,
                             symmetry=space.labels, enforce_kramers=False)
        probe.solve(ints)
        # ⚠ Built from the general run's own **sector weights**, not from its state labels: a
        # degenerate block spanning two conjugate sectors is one state of each, and counting
        # its printed name once per member asks for twice the states that exist.
        table = probe._sectors
        totals = np.rint(table.sector_weights(probe.last.vectors).sum(axis=0)).astype(int)
        request = {table.name(t): int(c) for t, c in zip(table.sectors, totals) if c > 0}
        per_irrep = _time_solve(
            lambda: FullCISolver(space.spaces.n_active, space.n_elec, n_states=request,
                                 symmetry=space.labels, enforce_kramers=False), ints)
        breach_h, breach_eri = sector_violation(h, eri, space.labels.labels, group.moduli)
        row = {
            "system": key, "n_active": n_active, "n_elec": space.n_elec,
            "breach_1e": breach_h, "breach_2e": breach_eri,
            "ndet": int(probe.ndet), "group": group.name,
            "n_states": n_states, "request": request,
            "dense": bool(probe.last.dense),
            "general_cpu_s": general[0], "general_n_apply": general[1],
            "sector_cpu_s": per_irrep[0], "sector_n_apply": per_irrep[1],
            "energy_agreement": abs(general[2] - per_irrep[2]),
            "speedup_cpu": general[0] / max(per_irrep[0], 1e-12),
            "speedup_apply": general[1] / max(per_irrep[1], 1),
        }
        cases.append(row)
        out(row)
    return cases


def measure_s6(out, budget_end):
    """The sub-manifold question with symmetry on: boundary gap and ensemble invariance."""
    ref = _reference(systems.get("ticl3"))
    space = active_space_for(ref, character=("Ti", "d"), n_active=10)
    ints = _integrals(ref, space)
    group = space.labels.group
    fermion = [group.irrep_name(t) for t in group.labels(fermion=True)]
    c_act = ref.spinors_in_ao()[:, space.spaces.active]
    spin_ao = spin_operator(np.asarray(ref.data.s_ao))
    spin_mo = np.stack([c_act.conj().T @ spin_ao[k] @ c_act for k in range(3)])

    rows = []
    plans = [("lowest 2 (ground doublet)", 2, None),
             ("lowest 4", 4, None),
             ("lowest 10 (whole d shell)", 10, None),
             ("per irrep, 1 each", None, {n: 1 for n in fermion}),
             ("per irrep, 2 each", None, {n: 2 for n in fermion}),
             ("per irrep, 5 each", None, {n: 5 for n in fermion})]
    for label, n_states, request in plans:
        if time.time() > budget_end:
            out({"note": "S6 stopped on the wall budget", "done": len(rows)})
            break
        kwargs = dict(symmetry=space.labels)
        if request is None:
            solver = FullCISolver(space.spaces.n_active, space.n_elec, n_states=n_states,
                                  **kwargs)
        else:
            solver = FullCISolver(space.spaces.n_active, space.n_elec, n_states=request,
                                  enforce_kramers=False, **kwargs)
        result = solver.casci(ints)
        report = state_average_boundary(solver, ints, margin=4, where="guess orbitals",
                                        spin_mo=spin_mo, gamma=result.gamma)
        rows.append({
            "selection": label, "n_states": solver.n_states,
            "irreps": list(result.irreps),
            "boundary_gap_cm": report.gap_cm,
            "binding_sector": report.sector,
            "spin_noninvariance": report.spin_noninvariance,
            "relative_cm": [float(x) for x in result.excitation_energies_cm()],
        })
        out(rows[-1])
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=float, default=540.0,
                        help="hard wall budget in seconds (the ten-minute rule)")
    parser.add_argument("--out", default="temp/symmetry_sectors.jsonl")
    args = parser.parse_args()
    set_verbosity(logging.WARNING)

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a")

    def emit(record):
        record = dict(record)
        record["t"] = round(time.time() - start, 2)
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        print(json.dumps(record, sort_keys=True))

    start = time.time()
    budget_end = start + args.budget
    emit({"stage": "S4"})
    measure_s4(emit, budget_end)
    emit({"stage": "S6"})
    measure_s6(emit, budget_end)
    emit({"stage": "done", "wall_s": time.time() - start, "cpu_s": _cpu()})
    handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
