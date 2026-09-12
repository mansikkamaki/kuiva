"""What an energy window costs and how well its pilot guesses, on Tier-3-shaped problems.

⚠ **A study generator, not a reference generator.** Everything it writes lands in ``temp/``;
nothing here is committed reference data and nothing in the test suite reads it.

The two questions the feature left open, both of which need systems the ten-minute ad-hoc
budget cannot reach (this run is explicitly authorized):

**P — how good is the pilot?** With no cheap CI upstream, a windowed network run takes its
ladder's first rung from a short campaign at a small bond dimension. On the small systems it
has been measured on it lands on the final count exactly and costs a fraction of a second;
both are properties of small systems with large gaps. Here the pilot's spectrum is solved
**once per system** and the rule is then applied to it at a grid of cutoffs, against the
count the same rule reads off a reference spectrum — the exact CI where one exists, and the
production-cap ladder where none does.

**B — do the generic starting vectors change what the fixed-count boundary diagnostic
reports?** The window's ladder asks the eigensolver for generic vectors at every rung because
a rung wants more roots than its warm start has vectors. The ordinary boundary diagnostic
solves ``n_states + 8`` roots from the same warm start and has exactly the same exposure; it
is advisory, so nothing has been wrong, but the two are the same measurement made two ways
and only one of them is protected. Stage ``boundary`` runs both at a converged CASSCF's own
orbitals and compares the gap and the cost.

**W — is cold growth the right default?** A rung that grows rebuilds its ensemble cold, as
the manifold loop does. The warm variant (``pad_roots``: keep the incumbent shared basis,
append random centers) is implemented and off. This measures both on the same system, the
same cutoff and the same cap, on **sweeps to convergence and CPU seconds** — and on the one
thing warm growth may not change, the resolved count.

The systems are the bridges of the Tier-3 road, taken from their **cached active integrals**
(``temp/dmrg_campaign/*_active_integrals.npz``, written by the Phase-2 stages): the coupled
dimer ``ti2cl6`` (CAS(2, 20), two sites of ten), the far trimer ``ti3f9_far`` (CAS(3, 30),
three sites of ten) and, for P only, the first Tier-3 system ``mn3_linear`` (CAS(15, 30)),
which has no oracle at all. Fixed integrals throughout: a window resolves at fixed orbitals,
so orbital quality is not an axis here.

Run discipline (the ad-hoc rules, which authorization does not suspend): one JSON record per
point written as it completes, a hard wall budget checked between points and enforced inside
the run rather than by an external timeout, a heartbeat line per point, and cheapest points
first so a budget kill leaves the interesting end measured. Judge progress from the heartbeat
file and ``/proc/<pid>``, never by matching a command line.

Usage::

    python tests/generate/dmrg_window_study.py --stage pilot    --budget 3600
    python tests/generate/dmrg_window_study.py --stage growth   --budget 5400
    python tests/generate/dmrg_window_study.py --stage boundary --budget 5400
    python tests/generate/dmrg_window_study.py --summary   # --regrade re-reads the rule

Records: ``temp/dmrg_window_study/<stage>.json``; log: ``temp/dmrg_window_study/<stage>.log``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dmrg_campaign as camp                                              # noqa: E402
from progress import Heartbeat                                            # noqa: E402

SCHEMA = 1
WORK = REPO / "temp/dmrg_window_study"

#: The systems, their cached integral file and the node partition each is measured on.
#: ``modes_per_node`` is the site-blocked chunking the bridge stage measured as the finest
#: that holds its ensembles; ``cap`` is the bond dimension the study runs at, taken from the
#: bridge ladder's own cost record (a point at that cap is tens of CPU seconds, not hours).
SYSTEMS: Dict[str, Dict] = {
    "ti2cl6": dict(file="ti2cl6_active_integrals.npz", cap=8, pilot_cap=8, exact_roots=24,
                   label="coupled dimer, CAS(2, 20)"),
    "ti3f9_far": dict(file="ti3f9_far_active_integrals.npz", cap=8, pilot_cap=8,
                      exact_roots=16, label="far trimer, CAS(3, 30)"),
    # ⚠ No oracle: the Tier-3 system's exact CI does not exist (1.6e8 determinants), so its
    # reference count comes from the network at a larger cap under a bounded sweep budget and
    # is a *measurement*, not a target.
    # ⚠ ``reference=None``: on this system there is nothing to compare the pilot's count
    # against. The exact CI does not exist (1.6e8 determinants) and a 16-root solve at a
    # larger cap does not fit the development box — measured, by being killed for memory.
    # What the Tier-3 leg measures is therefore the pilot's **cost**, which is the other half
    # of the question, and the record says the count has no target rather than inventing one.
    "mn3_linear": dict(file="mn3_linear_tier3_integrals.npz", cap=16, pilot_cap=8,
                       exact_roots=0, reference=None,
                       label="Tier 3: Mn(II) trimer, CAS(15, 30)"),
}

#: Cutoffs the rule is applied at [cm^-1], per system, chosen from that system's own
#: measured spectrum so that each one resolves to a **different** count — a grid of round
#: numbers on a spectrum with two clusters measures the same two counts five times. The
#: dimer's exact spectrum is 0 | 437 x3 | 12 264-12 338 x4 | 13 159-13 265 x4 | 17 994+,
#: so these resolve to 1, 4, 8, 12 and 16; the trimer's eight product states are degenerate
#: to under a wavenumber, so its grid starts above them.
CUTOFFS_CM: Dict[str, Tuple[float, ...]] = {
    "ti2cl6": (1.0, 500.0, 13000.0, 14000.0, 20000.0),
    # ⚠ Two distinct counts is all this system HAS: its eight product states are degenerate
    # to under a wavenumber and the next eight sit 15 316 cm^-1 up. A finer grid would
    # measure the same two counts five times.
    "ti3f9_far": (1.0, 5000.0, 20000.0),
    "mn3_linear": (1.0, 50.0, 500.0, 2000.0, 10000.0),
}

#: The ladder's first rung is forced low in the growth stage, so the ladder actually climbs;
#: this is the value ``EnergyWindow(initial=)`` is given there.
GROWTH_INITIAL = 2
#: Per-point wall budget inside a stage [s]. A point that exceeds it is recorded as
#: ``budget`` with what it had, never silently dropped.
POINT_BUDGET_S = 1800.0
#: Sweeps a resolution rung is allowed. The bridge stage's own number.
MAX_SWEEPS = 30
CONV_TOL = 1.0e-7
DAVIDSON_TOL = 1.0e-8


# --- plumbing -------------------------------------------------------------------------------
def _setup_logging(stage: str) -> Path:
    WORK.mkdir(parents=True, exist_ok=True)
    path = WORK / "{}.log".format(stage)
    root = logging.getLogger("kuiva")
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.FileHandler(path, mode="a")
    handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(handler)
    root.propagate = False
    return path


class Record:
    """One JSON file per stage, rewritten after every point (a kill leaves the rest)."""

    def __init__(self, stage: str, meta: Dict) -> None:
        self.path = WORK / "{}.json".format(stage)
        self.data: Dict = {"schema": SCHEMA, "stage": stage, "meta": meta, "points": []}
        if self.path.exists():
            try:
                old = json.loads(self.path.read_text())
                if old.get("schema") == SCHEMA and old.get("stage") == stage:
                    self.data["points"] = old.get("points", [])
            except Exception:                                   # noqa: BLE001 - a study file
                pass
        self.flush()

    def done(self, key: str) -> bool:
        return any(p.get("point_key") == key for p in self.data["points"])

    def add(self, point: Dict) -> None:
        self.data["points"] = [p for p in self.data["points"]
                               if p.get("point_key") != point.get("point_key")]
        self.data["points"].append(point)
        self.flush()

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, sort_keys=True, default=_jsonable))
        tmp.replace(self.path)


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


class ActiveIntegrals:
    """The duck-typed ``CASIntegrals`` surface the network layer reads, from the cache."""

    def __init__(self, path: Path) -> None:
        data = np.load(path)
        self._h = np.ascontiguousarray(data["h"])
        self._eri = np.ascontiguousarray(data["eri"])
        self.e_core = float(data["e_core"])
        self.n_elec = int(data["n_elec"])
        self.sites = [int(x) for x in data["sites"]]
        self.e_ci = (np.asarray(data["e_ci"], dtype=float) if "e_ci" in data.files
                     else np.zeros(0))

    def h_active_effective(self):
        return self._h

    def active_eri(self):
        return self._eri


def _load(key: str, n_roots: int) -> Tuple[ActiveIntegrals, object, List[int]]:
    """The cached integrals and the **site-blocked chain the bridge stage itself uses**.

    ⚠ The node partition is the campaign's own :func:`dmrg_phase2.bridge_partition` — the
    finest site-blocked chunking whose every two-site window holds ``n_roots`` — and not a
    round number chosen here. A coarser partition is not a free choice on these systems: ten
    modes per node is the three-node compile the campaign measured at 13 GB on the 30-spinor
    operator, and this study walked into it once (the Tier-3 point died of it) before using
    the function that already knew.
    """
    import dmrg_phase2 as p2

    spec = SYSTEMS[key]
    path = camp.WORK / spec["file"]
    if not path.is_file():
        raise SystemExit("no cached active integrals at {} — run the Phase-2 stage that "
                         "writes them first".format(path))
    ints = ActiveIntegrals(path)
    n_sites = int(max(ints.sites)) + 1
    site_sizes = [sum(1 for s in ints.sites if s == k) for k in range(n_sites)]
    sizes = p2.bridge_partition(site_sizes, ints.n_elec, int(n_roots))
    graph, _ = p2.bridge_graph(site_sizes, sizes)
    return ints, graph, [int(x) for x in sizes]


def _ttno(ints: ActiveIntegrals, graph):
    from kuiva.dmrg import TTNOTemplate

    return TTNOTemplate(graph).fill(ints.h_active_effective(), ints.active_eri())


def _rel_cm(e: Sequence[float]) -> List[float]:
    from kuiva.util.units import HARTREE_TO_CM

    e = np.asarray(e, dtype=float)
    return [round(float(x) * HARTREE_TO_CM, 4) for x in (e - e[0])]


def _window(cutoff_cm: float, **kw):
    from kuiva.util.window import EnergyWindow

    return EnergyWindow(float(cutoff_cm), max_states=64, **kw)


def _rule_count(relative_cm: Sequence[float], cutoff_cm: float) -> Dict:
    """The rule's verdict on a spectrum that is already solved — no solver involved.

    ⚠ ``relative_cm`` is in **wavenumbers** and the rule takes **hartree**: handing it the
    cm^-1 numbers makes every state astronomically far above any cutoff and the verdict is
    then 1 at every cutoff, which is what the first run of this stage recorded.
    """
    from kuiva.util.units import HARTREE_TO_CM
    from kuiva.util.window import resolve_window

    energies = np.asarray(relative_cm, dtype=float) / HARTREE_TO_CM
    v = resolve_window(energies, _window(cutoff_cm))
    return {"count": int(v.count), "complete": bool(v.complete),
            "witness_gap_cm": (None if v.witness_gap_cm is None
                               else round(float(v.witness_gap_cm), 3))}


# --- stage P: the pilot's estimate quality ---------------------------------------------------
def stage_pilot(record: Record, heartbeat: Heartbeat, *, budget: float) -> None:
    """One pilot campaign and one reference spectrum per system; the rule on both."""
    from kuiva.dmrg.window import PILOT_ROOTS, _solve_roots, sector_upper_bound
    from kuiva.dmrg.plan import two_site_capacity
    from kuiva.util import resources as res

    log = logging.getLogger("kuiva.window_study")
    for i, key in enumerate(SYSTEMS):
        if heartbeat.elapsed > budget:
            log.warning("*** budget spent before %s", key)
            return
        if record.done("pilot:" + key):
            log.info("   %s: already recorded", key)
            continue
        spec = SYSTEMS[key]
        res.clear()
        ints, graph, sizes = _load(key, PILOT_ROOTS)
        ttno = _ttno(ints, graph)
        capacity = two_site_capacity(graph, ints.n_elec)
        sector = sector_upper_bound(len(ints.sites), ints.n_elec)
        ceiling = max(1, min(sector, capacity))
        if ints.n_elec % 2 == 1 and ceiling % 2 == 1:
            ceiling -= 1
        roots = max(1, min(PILOT_ROOTS, ceiling))
        if ints.n_elec % 2 == 1 and roots % 2 == 1:
            roots = max(2, roots - 1)
        point: Dict = {"point_key": "pilot:" + key, "key": key, "label": spec["label"],
                       "node_sizes": sizes, "n_nodes": len(sizes),
                       "n_modes": len(ints.sites), "n_elec": ints.n_elec,
                       "capacity": int(capacity), "sector": int(sector),
                       "ceiling": int(ceiling), "pilot_roots": int(roots),
                       "pilot_cap": int(spec["pilot_cap"]), "cap": int(spec["cap"])}
        heartbeat.tick(i, system=key, stage="pilot")
        t0, c0 = time.time(), time.process_time()
        try:
            energies, _, sweeps, converged = _solve_roots(
                ttno, ints.n_elec, roots, max_bond=int(spec["pilot_cap"]),
                rng=np.random.default_rng(0), charge=None,
                solve_kwargs=dict(conv_tol=CONV_TOL, davidson_tol=DAVIDSON_TOL),
                memory_plan=False, max_sweeps=4)
            point["pilot"] = {"wall_s": round(time.time() - t0, 1),
                              "cpu_s": round(time.process_time() - c0, 1),
                              "n_sweeps": int(sweeps), "converged": bool(converged),
                              "relative_cm": _rel_cm(energies)}
        except Exception as exc:                                # noqa: BLE001 - an outcome
            point["pilot"] = {"status": "failed", "error": "{}".format(exc),
                              "cpu_s": round(time.process_time() - c0, 1)}
            record.add(point)
            log.warning("*** %s: the pilot failed: %s", key, exc)
            continue
        # The reference spectrum: the exact CI where the space allows it, otherwise the
        # production-cap network solve at the pilot's own root count — and nothing at all
        # where neither fits, which is a recorded outcome and not a failure.
        t0, c0 = time.time(), time.process_time()
        if "reference" in spec and spec["reference"] is None:
            point["reference"] = {"source": "none: no exact CI (1.6e8 determinants) and a "
                                            "16-root solve at a larger cap exceeds this "
                                            "machine's memory",
                                  "relative_cm": None}
            point["cutoffs"] = [{"cutoff_cm": c,
                                 "pilot": _rule_count(point["pilot"]["relative_cm"], c),
                                 "reference": None} for c in CUTOFFS_CM[key]]
            record.add(point)
            log.info("   %-11s pilot D=%d, %d roots, %d sweeps, %.1f CPU s; no reference "
                     "(counts %s)", key, point["pilot_cap"], point["pilot_roots"],
                     point["pilot"]["n_sweeps"], point["pilot"]["cpu_s"],
                     [r["pilot"]["count"] for r in point["cutoffs"]])
            continue
        if int(spec["exact_roots"]) > 0:
            ref = _exact_spectrum(ints, int(spec["exact_roots"]))
            ref["source"] = "exact CI"
        else:
            # ⚠ Bounded, and labelled so: a converged 16-root solve at the production cap on
            # a CAS(15, 30) Tier-3 system is hours, and what is compared is the *count* the
            # rule reads — a relative spectrum, which settles long before the energy does.
            sweep_budget = int(spec.get("ref_sweeps", MAX_SWEEPS))
            energies, _, sweeps, converged = _solve_roots(
                ttno, ints.n_elec, roots, max_bond=int(spec["cap"]),
                rng=np.random.default_rng(1), charge=None,
                solve_kwargs=dict(conv_tol=CONV_TOL, davidson_tol=DAVIDSON_TOL),
                memory_plan=False, max_sweeps=sweep_budget)
            ref = {"source": "network at D = {}, {} sweeps{}".format(
                       spec["cap"], sweep_budget,
                       "" if converged else " (bounded, not converged)"),
                   "relative_cm": _rel_cm(energies), "n_sweeps": int(sweeps),
                   "converged": bool(converged)}
        ref["wall_s"] = round(time.time() - t0, 1)
        ref["cpu_s"] = round(time.process_time() - c0, 1)
        point["reference"] = ref
        point["cutoffs"] = [
            {"cutoff_cm": c,
             "pilot": _rule_count(point["pilot"]["relative_cm"], c),
             "reference": _rule_count(ref["relative_cm"], c)}
            for c in CUTOFFS_CM[key]]
        record.add(point)
        _print_pilot(point)


def _exact_spectrum(ints: ActiveIntegrals, n_roots: int) -> Dict:
    """The lowest ``n_roots`` of the exact CI on these integrals."""
    from kuiva.mcscf.casci import FullCISolver

    solver = FullCISolver(len(ints.sites), ints.n_elec, n_states=n_roots,
                          enforce_kramers=False, on_split="warn")
    energies = solver.spectrum(np.ascontiguousarray(ints.h_active_effective()),
                               np.asarray(ints.active_eri()), int(n_roots))
    return {"relative_cm": _rel_cm(energies), "n_roots": int(n_roots),
            "ndet": int(solver.ndet)}


def _print_pilot(point: Dict) -> None:
    log = logging.getLogger("kuiva.window_study")
    pilot = point["pilot"]
    log.info("   %-11s pilot D=%d, %d roots, %d sweeps, %.1f CPU s; reference: %s",
             point["key"], point["pilot_cap"], point["pilot_roots"], pilot["n_sweeps"],
             pilot["cpu_s"], point["reference"]["source"])
    for row in point["cutoffs"]:
        log.info("      cutoff %9.1f cm^-1   pilot %3d%s   reference %3d%s",
                 row["cutoff_cm"], row["pilot"]["count"],
                 "" if row["pilot"]["complete"] else "+", row["reference"]["count"],
                 "" if row["reference"]["complete"] else "+")


# --- stage W: cold against warm growth -------------------------------------------------------
#: The cutoffs the growth stage runs, per system: chosen so the ladder has to climb from
#: ``GROWTH_INITIAL`` — i.e. the count is well above the first rung.
GROWTH_CUTOFFS: Dict[str, Tuple[float, ...]] = {
    # From each system's measured spectrum: a short climb (two rungs) and a long one
    # (four to six), because what warm growth can save is per *grown* rung.
    "ti2cl6": (500.0, 13000.0),
    "ti3f9_far": (1.0, 20000.0),
}


def stage_growth(record: Record, heartbeat: Heartbeat, *, budget: float) -> None:
    """The same ladder twice per (system, cutoff): cold growth and warm growth."""
    from kuiva.dmrg.window import PILOT_ROOTS, resolve_network_window
    from kuiva.util import resources as res

    log = logging.getLogger("kuiva.window_study")
    n = 0
    for key, cutoffs in GROWTH_CUTOFFS.items():
        spec = SYSTEMS[key]
        for cutoff in cutoffs:
            for warm in (False, True):
                n += 1
                tag = "growth:{}:{:.0f}:{}".format(key, cutoff, "warm" if warm else "cold")
                if record.done(tag):
                    log.info("   %s: already recorded", tag)
                    continue
                if heartbeat.elapsed > budget:
                    log.warning("*** budget spent before %s", tag)
                    return
                res.clear()
                ints, graph, sizes = _load(key, PILOT_ROOTS)
                ttno = _ttno(ints, graph)
                heartbeat.tick(n, system=key, cutoff=cutoff, warm=warm)
                point: Dict = {"point_key": tag, "key": key, "label": spec["label"],
                               "node_sizes": sizes, "cutoff_cm": cutoff, "warm": bool(warm),
                               "cap": int(spec["cap"]), "initial": GROWTH_INITIAL}
                t0, c0 = time.time(), time.process_time()
                try:
                    resolved = resolve_network_window(
                        ttno, ints.n_elec, _window(cutoff, initial=GROWTH_INITIAL),
                        max_bond=int(spec["cap"]), e_core=ints.e_core,
                        rng=np.random.default_rng(0), max_sweeps=MAX_SWEEPS,
                        solve_kwargs=dict(conv_tol=CONV_TOL, davidson_tol=DAVIDSON_TOL),
                        warm_growth=warm, pilot=False, report=False)
                except Exception as exc:                        # noqa: BLE001 - an outcome
                    point.update(status="failed", error="{}".format(exc),
                                 cpu_s=round(time.process_time() - c0, 1),
                                 wall_s=round(time.time() - t0, 1))
                    record.add(point)
                    log.warning("*** %s: %s", tag, exc)
                    continue
                res_ = resolved.resolution
                point.update(
                    status="ok", count=int(res_.count),
                    wall_s=round(time.time() - t0, 1),
                    cpu_s=round(time.process_time() - c0, 1),
                    rungs=[int(r.n_roots) for r in res_.rungs],
                    rung_cpu_s=[round(float(r.cpu_seconds), 2) for r in res_.rungs],
                    sweeps=list(resolved.sweeps),
                    witness_gap_cm=(None if res_.boundary_gap_cm is None
                                    else round(float(res_.boundary_gap_cm), 3)),
                    relative_cm=_rel_cm(res_.energies))
                record.add(point)
                log.info("   %-11s cutoff %7.0f  %-4s  count %3d  rungs %s  sweeps %s  "
                         "%.1f CPU s", key, cutoff, "warm" if warm else "cold",
                         point["count"], point["rungs"], point["sweeps"], point["cpu_s"])


# --- stage V: what a better pilot would cost --------------------------------------------------
#: ``(cap, sweeps)`` the pilot is re-run at, the first being today's default. The question is
#: not whether a longer pilot is more accurate — it is — but whether the extra cost buys a
#: *count*, which is the only thing a first rung is read for.
PILOT_VARIANTS = ((8, 4), (8, 12), (16, 4), (16, 12), (32, 12))
#: Systems the variants run on: the two with an exact spectrum to grade against.
PILOT_VARIANT_SYSTEMS = ("ti2cl6", "ti3f9_far")


def stage_pilotvar(record: Record, heartbeat: Heartbeat, *, budget: float) -> None:
    """The pilot at several ``(cap, sweeps)``, graded on the count it would hand the ladder."""
    from kuiva.dmrg.window import PILOT_ROOTS, _solve_roots
    from kuiva.util import resources as res

    log = logging.getLogger("kuiva.window_study")
    n = 0
    for key in PILOT_VARIANT_SYSTEMS:
        spec = SYSTEMS[key]
        reference = _reference_spectrum(record, key)
        if reference is None:
            log.warning("*** %s: no reference spectrum recorded; run --stage pilot first", key)
            continue
        for cap, sweeps in PILOT_VARIANTS:
            n += 1
            tag = "pilotvar:{}:{}:{}".format(key, cap, sweeps)
            if record.done(tag):
                continue
            if heartbeat.elapsed > budget:
                log.warning("*** budget spent before %s", tag)
                return
            res.clear()
            ints, graph, sizes = _load(key, PILOT_ROOTS)
            ttno = _ttno(ints, graph)
            heartbeat.tick(n, system=key, cap=cap, sweeps=sweeps)
            point: Dict = {"point_key": tag, "key": key, "cap": int(cap),
                           "sweeps_asked": int(sweeps), "node_sizes": sizes,
                           "roots": int(PILOT_ROOTS)}
            t0, c0 = time.time(), time.process_time()
            try:
                energies, _, done, converged = _solve_roots(
                    ttno, ints.n_elec, PILOT_ROOTS, max_bond=int(cap),
                    rng=np.random.default_rng(0), charge=None,
                    solve_kwargs=dict(conv_tol=CONV_TOL, davidson_tol=DAVIDSON_TOL),
                    memory_plan=False, max_sweeps=int(sweeps))
            except Exception as exc:                            # noqa: BLE001 - an outcome
                point.update(status="failed", error="{}".format(exc),
                             cpu_s=round(time.process_time() - c0, 1))
                record.add(point)
                log.warning("*** %s: %s", tag, exc)
                continue
            point.update(status="ok", n_sweeps=int(done), converged=bool(converged),
                         wall_s=round(time.time() - t0, 1),
                         cpu_s=round(time.process_time() - c0, 1),
                         relative_cm=_rel_cm(energies))
            point["cutoffs"] = [
                {"cutoff_cm": c, "pilot": _rule_count(point["relative_cm"], c),
                 "reference": _rule_count(reference, c)} for c in CUTOFFS_CM[key]]
            point["counts_match"] = all(r["pilot"]["count"] == r["reference"]["count"]
                                        for r in point["cutoffs"])
            record.add(point)
            log.info("   %-10s D=%-3d %2d sweeps (%2d run, %s): %6.1f CPU s, counts %s "
                     "against %s", key, cap, sweeps, point["n_sweeps"],
                     "converged" if converged else "bounded", point["cpu_s"],
                     [r["pilot"]["count"] for r in point["cutoffs"]],
                     [r["reference"]["count"] for r in point["cutoffs"]])


def _reference_spectrum(record: Record, key: str) -> Optional[List[float]]:
    """The reference relative spectrum recorded by the pilot stage, if it ran."""
    path = WORK / "pilot.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    for p in data.get("points", []):
        if p.get("key") == key and "reference" in p:
            return p["reference"]["relative_cm"]
    return None


# --- stage B: generic starting vectors for the fixed-count boundary diagnostic ---------------
#: The conventional-CI systems the boundary question is asked on, cheapest first. Each needs
#: a converged CASSCF checkpoint from the cost campaign (``temp/dmrg_campaign/<key>_casscf.h5``)
#: and a determinant space past the dense threshold — below it the eigensolver diagonalizes
#: the whole space and a starting vector is not a thing it has.
#: ⚠ Ordered by what each one costs, cheapest first, with the **load-bearing** case second
#: rather than last: the Dy(3+) free ion is the system whose 66-root solve converged 6 766
#: cm^-1 too high from a biased guess, which is the failure the generic vectors exist for.
#: ``fecl2_dd`` is the large-determinant case and goes last, where a budget stop costs least.
BOUNDARY_SYSTEMS = ("dycl3", "dy3p", "fecl2_dd")
#: Extra roots the diagnostic solves, i.e. the margin the CASSCF driver uses.
BOUNDARY_MARGIN = 8


def stage_boundary(record: Record, heartbeat: Heartbeat, *, budget: float) -> None:
    """The same boundary solve twice — today's warm-start-only guess, and with the generic
    vectors the window's ladder insists on — at a converged CASSCF's own orbitals.

    ⚠ What is compared is the **gap** the diagnostic reports and what the second solve costs.
    A generic solve that finds a *lower* extra root than the warm-started one would mean
    today's diagnostic reports a gap that is too large, i.e. an over-clean boundary line.
    """
    from kuiva.interface import api
    from kuiva.util import resources as res

    log = logging.getLogger("kuiva.window_study")
    for i, key in enumerate(BOUNDARY_SYSTEMS):
        if heartbeat.elapsed > budget:
            log.warning("*** budget spent before %s", key)
            return
        if record.done("boundary:" + key):
            log.info("   %s: already recorded", key)
            continue
        system = camp.get(key)
        ckpt = camp.orbital_checkpoint(system)
        if not ckpt.is_file():
            log.warning("*** %s: no converged CASSCF at %s", key, ckpt)
            continue
        res.clear()
        heartbeat.tick(i, system=key, stage="boundary")
        point: Dict = {"point_key": "boundary:" + key, "key": key}
        t0, c0 = time.time(), time.process_time()
        try:
            reference = camp.build_reference(system)
            outcome = api.casscf_from_checkpoint(reference, str(ckpt), boundary_check=0,
                                                 require_converged=False, report=False)
        except Exception as exc:                                # noqa: BLE001 - an outcome
            point.update(status="failed", error="{}".format(exc))
            record.add(point)
            log.warning("*** %s: %s", key, exc)
            continue
        solver = outcome.solver
        ints = camp.cas_integrals(reference, outcome.coeff, outcome.active)
        h = np.ascontiguousarray(ints.h_active_effective())
        eri = np.asarray(ints.active_eri())
        n = int(solver.n_states)
        point.update(n_states=n, ndet=int(solver.ndet), n_active=int(outcome.active.n_active),
                     n_active_elec=int(outcome.active.n_elec),
                     materialize_cpu_s=round(time.process_time() - c0, 1),
                     materialize_wall_s=round(time.time() - t0, 1))
        for label, generic in (("warm-only", None), ("generic", True)):
            t1, c1 = time.time(), time.process_time()
            try:
                e = solver.spectrum(h, eri, n + BOUNDARY_MARGIN, e_core=ints.e_core,
                                    generic=generic)
            except Exception as exc:                            # noqa: BLE001 - an outcome
                point[label] = {"status": "failed", "error": "{}".format(exc)}
                continue
            point[label] = {"gap_cm": round(float(_rel_cm(e[n - 1:n + 1])[1]), 4),
                            "cpu_s": round(time.process_time() - c1, 1),
                            "wall_s": round(time.time() - t1, 1),
                            "tail_cm": _rel_cm(e)[max(n - 2, 0):]}
        if "gap_cm" in point.get("warm-only", {}) and "gap_cm" in point.get("generic", {}):
            point["gap_change_cm"] = round(point["generic"]["gap_cm"]
                                           - point["warm-only"]["gap_cm"], 4)
            point["cpu_ratio"] = round(point["generic"]["cpu_s"]
                                       / max(point["warm-only"]["cpu_s"], 1e-9), 2)
        point["status"] = "ok"
        record.add(point)
        log.info("   %-10s %3d states of %6d dets: gap %10.3f -> %10.3f cm^-1 "
                 "(%.1f -> %.1f CPU s)", key, n, point["ndet"],
                 point["warm-only"]["gap_cm"], point["generic"]["gap_cm"],
                 point["warm-only"]["cpu_s"], point["generic"]["cpu_s"])


# --- the summary ------------------------------------------------------------------------------
def regrade() -> None:
    """Recompute every recorded point's counts from its stored spectra.

    The solves are the expensive part and the rule is free, so a mistake in the *grading* is
    repaired from the record rather than by re-running the campaign.
    """
    log = logging.getLogger("kuiva.window_study")
    for stage in ("pilot", "growth"):
        path = WORK / "{}.json".format(stage)
        if not path.is_file():
            continue
        data = json.loads(path.read_text())
        changed = 0
        for p in data["points"]:
            if stage != "pilot" or "pilot" not in p or "relative_cm" not in p["pilot"]:
                continue
            ref = p.get("reference", {}).get("relative_cm")
            rows = [{"cutoff_cm": c,
                     "pilot": _rule_count(p["pilot"]["relative_cm"], c),
                     "reference": None if ref is None else _rule_count(ref, c)}
                    for c in CUTOFFS_CM[p["key"]]]
            changed += int(rows != p.get("cutoffs"))
            p["cutoffs"] = rows
        path.write_text(json.dumps(data, indent=1, sort_keys=True, default=_jsonable))
        log.info("regraded %s: %d of %d points changed", stage, changed,
                 len(data["points"]))
        print("regraded {}: {} of {} points changed".format(stage, changed,
                                                            len(data["points"])))


def summarize() -> None:
    log = logging.getLogger("kuiva.window_study")
    for stage in ("pilot", "pilotvar", "growth", "boundary"):
        path = WORK / "{}.json".format(stage)
        if not path.is_file():
            continue
        data = json.loads(path.read_text())
        print("\n== {} ({} points)".format(stage, len(data["points"])))
        if stage == "pilot":
            print("{:<12} {:>9} {:>7} {:>7}  {:>9} {:>9} {}".format(
                "system", "cutoff", "pilot", "ref", "pilot CPU", "ref CPU", "reference"))
            for p in data["points"]:
                if "cutoffs" not in p:
                    print("{:<12} {}".format(p["key"], p.get("pilot", {}).get("error", "?")))
                    continue
                for row in p["cutoffs"]:
                    ref = row.get("reference")
                    print("{:<12} {:>9.1f} {:>7} {:>7}  {:>9.1f} {:>9} {}".format(
                        p["key"], row["cutoff_cm"], row["pilot"]["count"],
                        "-" if ref is None else ref["count"], p["pilot"]["cpu_s"],
                        "-" if ref is None else "{:.1f}".format(p["reference"]["cpu_s"]),
                        p["reference"]["source"][:40]))
        elif stage == "pilotvar":
            print("{:<12} {:>4} {:>7} {:>8} {:>9}  {:<22} {}".format(
                "system", "D", "sweeps", "CPU s", "converged", "pilot counts",
                "reference counts"))
            for p in data["points"]:
                if p.get("status") != "ok":
                    print("{:<12} {:>4} {}".format(p["key"], p["cap"],
                                                   p.get("error", "?")[:50]))
                    continue
                print("{:<12} {:>4} {:>7} {:>8.1f} {:>9}  {:<22} {}".format(
                    p["key"], p["cap"], p["n_sweeps"], p["cpu_s"],
                    str(p["converged"]),
                    str([r["pilot"]["count"] for r in p["cutoffs"]]),
                    [r["reference"]["count"] for r in p["cutoffs"]]))
        elif stage == "boundary":
            print("{:<12} {:>7} {:>8} {:>13} {:>13} {:>10} {:>9}".format(
                "system", "states", "ndet", "gap warm [cm]", "gap gen [cm]", "change",
                "CPU ratio"))
            for p in data["points"]:
                if p.get("status") != "ok":
                    print("{:<12} {}".format(p["key"], p.get("error", "?")[:60]))
                    continue
                print("{:<12} {:>7} {:>8} {:>13.3f} {:>13.3f} {:>10.4f} {:>9.2f}".format(
                    p["key"], p["n_states"], p["ndet"], p["warm-only"]["gap_cm"],
                    p["generic"]["gap_cm"], p["gap_change_cm"], p["cpu_ratio"]))
        else:
            print("{:<12} {:>8} {:>6} {:>6} {:>10} {:>8} {}".format(
                "system", "cutoff", "mode", "count", "CPU s", "sweeps", "rungs"))
            for p in data["points"]:
                if p.get("status") != "ok":
                    print("{:<12} {:>8.0f} {:>6} {}".format(
                        p["key"], p["cutoff_cm"], "warm" if p["warm"] else "cold",
                        p.get("error", "?")[:60]))
                    continue
                print("{:<12} {:>8.0f} {:>6} {:>6} {:>10.1f} {:>8} {}".format(
                    p["key"], p["cutoff_cm"], "warm" if p["warm"] else "cold", p["count"],
                    p["cpu_s"], sum(p["sweeps"]), p["rungs"]))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage",
                        choices=("pilot", "pilotvar", "growth", "boundary"))
    parser.add_argument("--budget", type=float, default=3600.0,
                        help="wall budget for the stage [s]")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--regrade", action="store_true",
                        help="recompute the counts from the stored spectra")
    args = parser.parse_args(argv)
    if args.regrade:
        _setup_logging("regrade")
        regrade()
        summarize()
        return 0
    if args.summary:
        summarize()
        return 0
    if args.stage is None:
        parser.error("--stage or --summary")
    log_path = _setup_logging(args.stage)
    log = logging.getLogger("kuiva.window_study")
    log.info("\n=== window study: stage %s, budget %.0f s ===", args.stage, args.budget)
    meta = {"stage": args.stage, "budget_s": args.budget,
            "kuiva_kernels": os.environ.get("KUIVA_KERNELS", "auto"),
            "threads": os.environ.get("KUIVA_NUM_THREADS", "unset"),
            "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    record = Record(args.stage, meta)
    heartbeat = Heartbeat("dmrg_window_" + args.stage, budget_seconds=args.budget,
                          meta=meta)
    t0 = time.time()
    if args.stage == "pilot":
        stage_pilot(record, heartbeat, budget=args.budget)
    elif args.stage == "pilotvar":
        stage_pilotvar(record, heartbeat, budget=args.budget)
    elif args.stage == "growth":
        stage_growth(record, heartbeat, budget=args.budget)
    else:
        stage_boundary(record, heartbeat, budget=args.budget)
    record.data["meta"]["elapsed_s"] = round(time.time() - t0, 1)
    record.flush()
    heartbeat.finish(stage=args.stage)
    print("record: {}\nlog: {}".format(record.path, log_path))
    summarize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
