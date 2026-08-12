"""The frozen-core measurement protocol for SC-NEVPT2.

The predeclared question
------------------------
**Does freezing the core change the target relative energies — SOC splittings, term energies —
by less than the method's own accuracy band, at a worthwhile cost saving?** The answer decides
the default of ``sc_nevpt2(frozen_core=...)``, and it is decided from measurement rather than
from what other programs do. The plausible outcome was written down in advance explicitly as a
*guess carrying no weight*, which is what this script exists to replace.

The protocol
------------
* **Systems**: the **p^4 series O / S / Se / Te**, defined below, plus any key from
  ``systems.py`` on request (``--only bi``, ``--only tlh``, ...).

  ⚠ **The series is local to this script and that is deliberate, not a lapse from the shared-systems rule's
  "one source of truth for systems".** That rule binds the validation *tiers*, whose reference
  values have to be reproducible by another program; this is a measurement, and what it needs
  is a family in which **only the depth of the core changes** — same term structure (3P + 1D +
  1S), same active space (the p shell), same state count, Z from 8 to 52. One heavy molecule
  cannot answer "how does the frozen-core error grow with the core" at all, and the ``systems.py``
  members that could (``bi``, ``tlh``, ``ce3p``) cost a state-averaged CASSCF of order ten
  CPU-minutes each — measured, and over the ten-minute ad-hoc budget for the sweep. They remain reachable by
  key for a longer run, and the record says which systems a given file contains.
* **Bases**: the project default *and* a core-valence set (``--basis cc-pwCVDZ-X2C``). ⚠ Both,
  because a frozen-core error measured in a valence basis is invisible for the wrong reason —
  the basis cannot correlate the core at all — and that is the design's explicit requirement.
* **Ladder**: all-electron -> deep core frozen -> semicore also frozen, each rung a **physical
  statement** (whole shells), never an index count. The thresholds are found from the
  pseudo-canonical ``eps`` spectrum itself: :func:`shell_ladder` places them in the largest
  gaps between degeneracy groups, so a rung always freezes whole shells and the record stores
  the energy that names each one.
* **Observables**: per-class ``E2`` shifts, the SOC spectrum in cm^-1 and how each multiplet
  moves, and cost as **CPU seconds** (never wall time on the dev box).
* **One state-averaged CASSCF per system, shared by every rung**, so a difference between rungs
  is the perturbation and nothing else.

⚠ **A CASSCF reference, not a CASCI at the SCF orbitals, and the first attempt at this script
got that wrong in a way worth recording.** ``Sir (0')`` is the only class whose perturber
carries a one-body coefficient — the inactive Fock element ``f^I_ai`` — and at a *converged*
CASSCF the generalized Fock element ``(F^I + F^A)_ai`` vanishes, which is what keeps it small.
At unoptimized orbitals it does not, and ``Sir`` then measures the **Brillouin violation of the
reference** rather than any correlation effect: on TlH at the scalar SCF orbitals it came out at
**-36.7 Eh** against -0.84 Eh for the whole MP2-like ``Sijrs`` bulk, and swamped every
frozen-core difference the run was there to measure. That is a useful diagnostic in its own
right — a huge ``Sir`` means the orbitals are not stationary — and it is a trap for any
measurement built on a cheap reference.

⚠ Two things this measurement cannot answer, stated up front
-------------------------------------------------------------
* **A valence basis cannot correlate the core at all**, so a small frozen-core error measured in
  ``x2c-SVPall-2c`` is a statement about the basis, not about the approximation. Running a
  rung in a core-valence set (``--basis cc-pwCVDZ-X2C``) is what makes the number mean
  something, and the record stores the basis for exactly that reason.
* **Nothing here says anything about a frozen-core CASSCF.** Only the perturbation's label
  ranges change; the reference wavefunction is untouched by construction, and the record
  asserts that by storing ``e_casscf`` per rung.

Cost and how this script is bounded
--------------------------------------------
Ordered cheapest first, incremental (the output file is rewritten after every rung, so a run
killed at the budget keeps what it finished), with a hard ``--budget`` checked between systems
and a :mod:`progress` heartbeat. The default sweep is designed to fit the ten-minute rule; the
4f pair is the expensive end and is simply not reached if the budget runs out, which the record
says rather than hides.

Usage::

    python tests/generate/nevpt2_frozen_core.py --only tlh --budget 120      # smoke
    python tests/generate/nevpt2_frozen_core.py --budget 540                 # the sweep
    python tests/generate/nevpt2_frozen_core.py --only bi --basis cc-pwCVDZ-X2C
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import systems as sysdef                                                    # noqa: E402
from progress import Heartbeat                                              # noqa: E402

REF_OUT = REPO / "tests/reference/nevpt2_frozen_core.json"

#: Schema of the emitted file. Bump when the *meaning* of a stored field changes.
SCHEMA = 1

#: Working-memory limit [GB].
MEMORY_GB = 8.0

#: Degeneracy grouping width for reading multiplets out of a spectrum [cm^-1].
GROUP_TOL_CM = 1.0

#: The p^4 series: one term structure (3P + 1D + 1S), one active space (the p shell), one state
#: count (15 = the whole active CI space, so the average is complete by construction), and a core
#: that goes from one shell to four. ⚠ Measurement-only systems — see the module docstring on why
#: they are not in ``systems.py``.
SERIES: Dict[str, sysdef.System] = {
    symbol.lower(): sysdef.System(
        key=symbol.lower(), label=symbol, atoms=[(symbol, (0.0, 0.0, 0.0))],
        charge=0, spin=2, basis="x2c-SVPall-2c", basis_matched="cc-pwCVDZ-X2C",
        # ⚠ active_l is deliberately EMPTY: the frontier window, not "the lowest p-character
        # pairs". Character selection takes the *lowest* Kramers pairs of that character, which
        # for Te is the 2p — a core shell (the AO-label trap in its most literal form). The frontier
        # window `[nelec - nelecas, ...)` is the valence p shell at every Z in this series.
        ncas=3, nelecas=4, active_l="", nroots={3: 3, 1: 5 + 1}, tier1=False,
        physics_note="p^4: 3P(9) + 1D(5) + 1S(1) = 15 spinor states = the complete active CI "
                     "space. Same structure at every Z, so the frozen-core error is measured "
                     "against core depth and nothing else")
    for symbol in ("O", "S", "Se", "Te")
}

#: Order the sweep runs in: cheapest first, so a structural failure surfaces in the first
#: minute rather than the last.
ORDER = ("o", "s", "se", "te")


def resolve(key: str) -> sysdef.System:
    """A measurement system from :data:`SERIES`, or any ``systems.py`` key."""
    return SERIES[key] if key in SERIES else sysdef.get(key)

#: How many frozen-core rungs to attempt above the all-electron one.
N_RUNGS = 2


def shell_ladder(eps: np.ndarray, n_rungs: int = N_RUNGS, rtol: float = 1e-9,
                 *, from_top: bool = False) -> List[float]:
    """Thresholds [Eh] placed in the ``n_rungs`` largest gaps of ``eps``.

    ⚠ **A shell boundary, never a count.** Degenerate groups are found first, so a threshold
    always falls *between* shells and can never cut one in half — which
    :func:`kuiva.pt.nevpt2.select_correlated` would refuse anyway, but a generator that
    produced refused inputs would be measuring nothing.

    ``from_top=False`` (the frozen core) returns them in ascending energy, so rung 1 freezes the
    deepest shells and rung 2 adds the next ones out. ``from_top=True`` (deleted virtuals)
    returns them descending, so rung 1 deletes the highest.

    ⚠ For the virtual side the "shells" are not shells: a virtual spectrum has no shell
    structure to speak of, only whatever gaps the basis happens to leave, so a threshold there
    is a **basis statement** and not a physical one. That is exactly why the record stores the
    threshold and the resulting count rather than a label.
    """
    eps = np.asarray(eps, dtype=float).ravel()
    if eps.size < 2:
        return []
    scale = max(float(np.max(np.abs(eps))), 1.0)
    edges = [i + 1 for i in range(eps.size - 1) if eps[i + 1] - eps[i] > rtol * scale]
    if not edges:
        return []
    gaps = sorted(edges, key=lambda i: eps[i] - eps[i - 1], reverse=True)
    chosen = sorted(gaps[:int(n_rungs)], reverse=from_top)
    return [0.5 * float(eps[i - 1] + eps[i]) for i in chosen]


def multiplets(energies: np.ndarray, tol_cm: float = GROUP_TOL_CM):
    """``[(degeneracy, barycentre [cm^-1]), ...]`` in energy order."""
    from kuiva.props.multiplet import HARTREE_TO_CM
    rel = np.sort((np.asarray(energies, float) - np.min(energies)) * HARTREE_TO_CM)
    groups: List[List[float]] = []
    for value in rel:
        if groups and value - groups[-1][0] <= tol_cm:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [(len(g), float(np.mean(g))) for g in groups]


#: CASSCF bounds. A budget, not an expectation: a record that hits ``MAX_ITER`` is
#: written with ``casscf_converged: false`` rather than silently trusted.
CONV_GRAD = 1.0e-4
MAX_ITER = 40


def run_system(system: sysdef.System, *, basis: Optional[str] = None,
               memory_gb: float = MEMORY_GB, n_rungs: int = N_RUNGS,
               max_iter: int = MAX_ITER, side: str = "core",
               heartbeat=None) -> Dict:
    """One system: the SA-CASSCF once, then the frozen-core ladder on top of it."""
    from kuiva.interface import api
    from kuiva.pt.classes import available_classes
    from kuiva.pt.nevpt2 import pseudo_canonicalize, sc_nevpt2
    from kuiva.util import resources as res

    # ⚠ Release the previous system's reservations: `resources.BUDGET` is process-global by
    # design, so in a batch driver it accumulates and a later, larger system is refused
    # against a limit its predecessors filled. Same idiom as `tier2_kuiva.run_system`.
    res.BUDGET.clear()
    basis_name = basis or system.basis
    if side not in ("core", "virtual"):
        raise SystemExit("side must be 'core' or 'virtual', got {!r}".format(side))
    record: Dict = {"key": system.key, "label": system.label, "basis": basis_name,
                    "charge": system.charge, "spin": system.spin,
                    "ncas": system.ncas, "nelecas": system.nelecas,
                    "n_states": system.soc_states, "screening": "none", "rungs": []}

    t0, c0 = time.time(), time.process_time()
    molecule = api.Molecule(atoms=system.atoms, basis=basis_name, charge=system.charge,
                            spin=system.spin)
    # screening="none": what is under measurement is the frozen-core approximation, and the
    # two-electron picture change costs a four-component atomic solve it cannot affect.
    reference = api.spinor_reference(molecule, memory_gb=memory_gb, screening="none")
    n_inactive = reference.data.nelec_total - system.nelecas
    if n_inactive % 2:
        raise SystemExit("{}: odd inactive spinor count".format(system.key))
    selection = (dict(character=([0], system.active_l), n_active=2 * system.ncas,
                      n_active_elec=system.nelecas) if system.active_l else
                 dict(active=list(range(n_inactive, n_inactive + 2 * system.ncas)),
                      n_active_elec=system.nelecas))
    outcome = api.casscf(reference, n_states=system.soc_states, mode="second-order",
                         conv_grad=CONV_GRAD, max_iter=max_iter, report=False, **selection)
    cas, spaces, coeff = outcome.ci, outcome.active.spaces, outcome.coeff
    record.update(nao=int(reference.data.nao), nspinor=int(reference.nspinor),
                  n_inactive=int(spaces.n_inactive), n_virtual=int(spaces.n_virtual),
                  active_space=outcome.active.description,
                  casscf_converged=bool(outcome.converged),
                  casscf_iterations=int(outcome.orbital.n_iterations),
                  casscf_grad_norm=float(outcome.orbital.grad_norm),
                  e_casscf_avg=float(outcome.energy),
                  reference_wall=time.time() - t0, reference_cpu=time.process_time() - c0)

    h_ao = reference.h_one_electron()
    canonical = pseudo_canonicalize(reference.factors, h_ao, coeff, spaces, cas.gamma,
                                    e_nuc=reference.data.e_nuc)
    record["eps_inactive"] = [float(e) for e in canonical.eps_inactive]
    record["side"] = side
    if side == "virtual":
        ladder: List[Optional[float]] = [None] + shell_ladder(
            canonical.eps_virtual, n_rungs, from_top=True)
    else:
        ladder = [None] + shell_ladder(canonical.eps_inactive, n_rungs)

    baseline = None
    for rung, threshold in enumerate(ladder):
        t, c = time.time(), time.process_time()
        kwargs = ({"deleted_virtual": threshold} if side == "virtual"
                  else {"frozen_core": threshold})
        corrected = sc_nevpt2(reference.factors, h_ao, coeff, spaces, cas.vectors,
                              system.nelecas, energies=cas.energies,
                              e_nuc=reference.data.e_nuc, report=False, **kwargs)
        levels = multiplets(corrected.total_energies)
        entry = {
            "rung": rung,
            "threshold_eh": threshold,
            "n_frozen": int(corrected.n_frozen + corrected.n_deleted),
            "e_casscf": float(corrected.e_casscf[0]),
            "e2": float(corrected.e2[0]),
            "e2_all_states": [float(x) for x in corrected.e2],
            "class_e2": {n: float(corrected.class_energies[n][0])
                         for n in available_classes()},
            "levels_cm": [[int(d), round(b, 4)] for d, b in levels],
            "wall_seconds": time.time() - t,
            "cpu_seconds": time.process_time() - c,
        }
        if baseline is None:
            baseline = entry
        else:
            entry["de2_eh"] = entry["e2"] - baseline["e2"]
            entry["dlevels_cm"] = [round(b - b0, 4) for (_, b), (_, b0)
                                   in zip(levels, [(d, b) for d, b in baseline["levels_cm"]])]
            entry["worst_level_shift_cm"] = (max(abs(x) for x in entry["dlevels_cm"])
                                             if entry["dlevels_cm"] else 0.0)
            entry["cpu_saving"] = 1.0 - entry["cpu_seconds"] / baseline["cpu_seconds"]
        record["rungs"].append(entry)
        if heartbeat is not None:
            heartbeat.tick(rung, system=system.key, stage="frozen-core",
                           energy=entry["e2"])
    record["wall_seconds"] = time.time() - t0
    record["cpu_seconds"] = time.process_time() - c0
    return record


def report(record: Dict) -> None:
    print("{}  {}  [{}]  {} states, {} inactive / {} virtual spinors".format(
        record["label"], record["basis"], record.get("side", "core"),
        record["n_states"], record["n_inactive"], record["n_virtual"]))
    print("  {:>5s} {:>14s} {:>8s} {:>18s} {:>14s} {:>16s} {:>10s}".format(
        "rung", "threshold[Eh]", "removed", "E2 [Eh]", "dE2 [Eh]", "worst dE [cm^-1]",
        "cpu [s]"))
    for entry in record["rungs"]:
        print("  {:>5d} {:>14s} {:>8d} {:> 18.10f} {:>14s} {:>16s} {:>10.1f}".format(
            entry["rung"],
            "all-electron" if entry["threshold_eh"] is None
            else "{:.4f}".format(entry["threshold_eh"]),
            entry["n_frozen"], entry["e2"],
            "-" if "de2_eh" not in entry else "{:.3e}".format(entry["de2_eh"]),
            "-" if "worst_level_shift_cm" not in entry
            else "{:.4f}".format(entry["worst_level_shift_cm"]),
            entry["cpu_seconds"]))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--basis", default=None, help="override the system's basis")
    parser.add_argument("--memory-gb", type=float, default=MEMORY_GB)
    parser.add_argument("--rungs", type=int, default=N_RUNGS)
    parser.add_argument("--max-iter", type=int, default=MAX_ITER)
    parser.add_argument("--side", choices=("core", "virtual"), default="core",
                        help="freeze the core from below, or delete virtuals from above")
    parser.add_argument("--budget", type=float, default=540.0,
                        help="hard wall budget [s], checked between systems (12.0)")
    parser.add_argument("--merge", action="store_true",
                        help="keep records already in the output file")
    args = parser.parse_args(argv)

    keys = list(args.only) if args.only else list(ORDER)
    records: List[Dict] = []
    if args.merge and REF_OUT.exists():
        records = json.loads(REF_OUT.read_text()).get("records", [])
    beat = Heartbeat("nevpt2_frozen_core")
    start = time.time()
    skipped: List[str] = []
    for key in keys:
        if time.time() - start > args.budget:
            skipped.append(key)
            continue
        record = run_system(resolve(key), basis=args.basis, memory_gb=args.memory_gb,
                            n_rungs=args.rungs, max_iter=args.max_iter, side=args.side,
                            heartbeat=beat)
        report(record)
        records = [r for r in records
                   if not (r["key"] == record["key"] and r["basis"] == record["basis"]
                           and r.get("side", "core") == record["side"])]
        records.append(record)
        # Incremental: a run killed at the budget still yields what it finished.
        REF_OUT.parent.mkdir(parents=True, exist_ok=True)
        REF_OUT.write_text(json.dumps(
            {"schema": SCHEMA, "not_measured": skipped, "records": records},
            indent=1, sort_keys=True) + "\n")
    if skipped:
        print("NOT MEASURED (budget {:.0f} s exhausted): {}".format(args.budget,
                                                                    ", ".join(skipped)))
    print("wrote {}".format(REF_OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
