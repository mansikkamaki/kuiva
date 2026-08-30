"""How accuracy deteriorates when the tensor network is a genuine approximation, and what
the cheapest settings that still answer the physical question cost.

⚠ **A study generator, not a reference generator.** Everything it writes lands in ``temp/``;
nothing here is committed reference data and nothing in the test suite reads it.

The question
------------
Every tensor-network number validated so far was taken at a *saturating* bond dimension,
where the solver is exact CI wearing DMRG clothes. That says the implementation is right and
says nothing about the approximation. This measures the other regime: a binding bond cap, a
fixed sweep budget, real heavy-element multiplet structure — and grades the result against
the exact CI on the **same integrals and the same orbitals**, on a three-level scale.

*Quantitative* means interchangeable with the exact answer. *Qualitative* means visibly
different numbers and the same physics, by an amount comparable to what is already neglected
outside the active space. *Unacceptable* means the answer to the physical question changed.
The deliverable per system is **two numbers, not one**: the smallest (D, sweeps) reaching
each tier — the second being the fit-for-purpose floor of a fast calculation — plus the
CPU-seconds ratio to the exact solve at the same task.

Protocol
--------
**Fixed orbitals throughout this file.** One converged state-averaged CASSCF per system
supplies the orbitals; every ladder point is a network CASCI on those, against an exact
CASCI on those. Orbital quality is a separate axis and belongs to a separate protocol —
mixing them would compare optimizer trajectories rather than CI methods. Two stated
topologies (a Fiedler-ordered chain and a mutual-information tree); topology *discovery* is
not load-bearing anywhere here.

Run discipline
--------------
Hours, not minutes, and designed accordingly: one JSON record per (system, topology, cap)
written as it completes, a hard wall budget checked between points, a heartbeat line per
point, caps run cheapest-first so a budget kill leaves the *interesting* low-cap end
measured, and every refusal or non-convergence recorded as an outcome rather than raised.
Judge progress from the heartbeat file and ``/proc/<pid>``, never by matching a command line.

Usage::

    python tests/generate/dmrg_cost_ladder.py --stage s0.1                # smoke, < 10 min
    python tests/generate/dmrg_cost_ladder.py --stage s0.2 --budget 10800
    python tests/generate/dmrg_cost_ladder.py --stage s1.2 --budget 10800

Records: ``temp/dmrg_cost_ladder/<stage>.json``; log: ``temp/dmrg_cost_ladder/<stage>.log``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dmrg_campaign as camp                                              # noqa: E402
from progress import Heartbeat                                            # noqa: E402

SCHEMA = 1

#: What each stage does. ``jobs`` are ``(system key, topologies)``; a stage with no jobs is
#: a preparation stage handled by its own function. Ordered so that a stage only ever
#: depends on one that ran before it — the orbitals are the dependency, and they are
#: checkpointed, so a re-run of a later stage never repays an earlier one.
STAGES: Dict[str, Dict] = {
    "s0.1": dict(kind="ladder", label="scaffolding smoke test",
                 jobs=(("tif3", ("path",)),), caps=(2, 4, 8), budget=540.0),
    "s0.2": dict(kind="front-end", label="atomic mean field warm + SCF checks",
                 jobs=(("dycl3", ()), ("uf3", ())), budget=3.0 * 3600),
    "s0.3": dict(kind="orbitals", label="reference state-averaged CASSCFs",
                 jobs=(("dycl3", ()), ("uf3", ())), budget=3.0 * 3600),
    "s1.1": dict(kind="ladder", label="transition-metal ladders",
                 jobs=(("tif3", ("path", "tree")), ("fecl2", ("path", "tree"))),
                 budget=2.0 * 3600),
    "s1.2": dict(kind="ladder", label="DyCl3 ladder, chain topology",
                 jobs=(("dycl3", ("path",)),), budget=3.0 * 3600),
    "s1.3": dict(kind="ladder", label="DyCl3 tree topology + the free-ion control",
                 jobs=(("dycl3", ("tree",)), ("dy3p", ("path", "tree"))),
                 budget=3.0 * 3600),
    "s1.4": dict(kind="ladder", label="UF3 ladder, both topologies",
                 jobs=(("uf3", ("path", "tree")),), budget=2.0 * 3600),
}


# --- output plumbing ------------------------------------------------------------------------
def _setup_logging(stage: str) -> Path:
    """Kuiva's own output to a file at INFO, the console to WARNING.

    The output stream IS the output file, and a campaign's evidence — the boundary
    diagnostics, the truncation refusals, the picture-change statement on every property
    matrix — lives in it. What the console needs is only what went wrong.
    """
    from kuiva.util import logging as klog

    camp.RECORDS.mkdir(parents=True, exist_ok=True)
    path = camp.RECORDS / "{}.log".format(stage)
    klog.set_verbosity("INFO")
    root = logging.getLogger("kuiva")
    for handler in root.handlers:
        if isinstance(handler, logging.StreamHandler) \
                and not isinstance(handler, logging.FileHandler):
            handler.setLevel(logging.WARNING)
    klog.add_file_handler(path, level=logging.INFO)
    return path


class Record:
    """The stage's JSON record, rewritten after every completed unit of work.

    ⚠ ``resume`` keeps what a previous invocation of this stage already measured and adds
    to it. A ladder point costs tens of minutes and a stage costs hours, so a run that is
    interrupted — by a wall budget, a machine, or anything else — must not throw away the
    points it had already paid for. Nothing is recomputed and nothing is overwritten; a
    point already present for a given ``(system, topology, cap)`` is skipped, whatever its
    outcome, because a refusal and a non-convergence are results too.
    """

    def __init__(self, stage: str, path: Path, meta: Dict, resume: bool = False) -> None:
        self.path = path
        previous: Dict = {}
        if resume and path.is_file():
            try:
                previous = json.loads(path.read_text())
            except ValueError:
                previous = {}
        self.data: Dict = dict(schema=SCHEMA,
                               generator="tests/generate/dmrg_cost_ladder.py",
                               stage=stage, **meta)
        self.data["jobs"] = previous.get("jobs", [])
        self.data["points"] = previous.get("points", [])
        if previous:
            self.data["resumed_from"] = previous.get("elapsed_s")
        self.flush()

    def done_points(self) -> set:
        """``(key, topology, cap)`` already measured, so a resume does not repeat them."""
        return {(p.get("key"), p.get("topology"), p.get("cap"))
                for p in self.data["points"]
                if p.get("cap") is not None and p.get("status") not in (None, "skipped")}

    def has_job(self, key: str) -> bool:
        return any(j.get("key") == key for j in self.data["jobs"])

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.part")
        with open(tmp, "w") as fh:
            json.dump(self.data, fh, indent=1, sort_keys=True, default=_jsonable)
        os.replace(tmp, self.path)

    def add_job(self, job: Dict) -> None:
        self.data["jobs"].append(job)
        self.flush()

    def add_point(self, point: Dict) -> None:
        self.data["points"].append(point)
        self.flush()


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _env_meta() -> Dict:
    import kuiva
    from kuiva.ci import kernels
    from kuiva.util import native

    # ⚠ Activate before asking: the import is lazy by design, so `available()` on its own
    # reports False in a process that has not yet dispatched a kernel — which would record
    # every run of this campaign as a pure-NumPy one whatever it actually ran on. The
    # backend is provenance here, not a variable, and provenance that is silently wrong is
    # worse than none.
    native.activate()
    return {"kuiva_version": kuiva.__version__,
            "kernel_backend": native.fingerprint_token(),
            "preferred_backend": kernels.preferred_backend(),
            "kuiva_num_threads": os.environ.get("KUIVA_NUM_THREADS"),
            "kuiva_kernels": os.environ.get("KUIVA_KERNELS", "auto"),
            "native_available": bool(native.available()),
            "native_build": native.build_id() if native.available() else None,
            "bands": camp.BANDS.__dict__,
            "block_tol_cm": camp.BLOCK_TOL_CM}


# --- the stages -----------------------------------------------------------------------------
def stage_front_end(key: str, record: Record, heartbeat) -> Dict:
    """Build the reference and stop there: the four-component atomic solves and the SCF.

    ⚠ An unconverged SCF is recorded as a **failure of this stage**, never proceeded past.
    Everything downstream is built on those orbitals and "the CASSCF re-optimizes them" is a
    hope rather than a property; the fix is the guess, not a flag.
    """
    system = camp.get(key)
    t0, c0 = time.time(), time.process_time()
    reference = camp.build_reference(system)
    job = {"key": key, "stage_kind": "front-end", "label": system.label,
           "basis": system.basis, "charge": system.charge, "spin": system.spin,
           "geometry_note": system.geom_note, "physics_note": system.physics_note,
           "protocol_note": system.protocol_note,
           "atoms": [[s, list(xyz)] for s, xyz in system.atoms],
           "nao": int(reference.data.nao),
           "nspinor": int(reference.nspinor),
           "e_scf": float(reference.data.e_scf),
           "scf_converged": bool(reference.data.converged),
           "n_cholesky": int(reference.factors.naux),
           "wall_s": round(time.time() - t0, 1),
           "cpu_s": round(time.process_time() - c0, 1)}
    if reference.data.soc is not None:
        job["hamiltonian"] = reference.data.soc.provenance()
    # The active-space statement is trap-checked here rather than at the first ladder point:
    # a selection that landed on the wrong shell is silent in every observable this
    # campaign grades, and the cheapest place to catch it is before anything expensive runs.
    space = _resolve_space(reference, system)
    job["active_space"] = space.description
    job["active_indices"] = [int(i) for i in space.spaces.active]
    heartbeat.tick(0, system=key, stage="front-end",
                   converged=int(reference.data.converged))
    record.add_job(job)
    if not reference.data.converged:
        raise RuntimeError("{}: the scalar SCF did not converge; fix the guess before "
                           "any stage downstream of it runs".format(key))
    return job


def _resolve_space(reference, system):
    from kuiva.interface import api
    return api.active_space_for(reference, **system.selection())


def stage_orbitals(key: str, record: Record, heartbeat, *, deadline: float) -> Dict:
    """The reference state-averaged CASSCF whose orbitals every ladder point reuses.

    Checkpointed and resumed from its own checkpoint, so an interrupted campaign never
    repays an orbital optimization. Both state-average boundary diagnostics and the
    spin-invariance figure are recorded whatever they say — the starting-orbital one is the
    statement about whether the *trajectory* was safe, and it is the one that cannot be
    recovered afterwards.
    """
    system = camp.get(key)
    reference = camp.build_reference(system)
    coeff, space, rec = camp.converged_orbitals(
        system, reference, deadline=max(0.0, deadline - time.time() - 120.0))
    job = {"key": key, "stage_kind": "orbitals", "label": system.label,
           "n_states": system.n_states, "casscf_mode": system.casscf_mode,
           "checkpoint": str(camp.orbital_checkpoint(system)),
           "protocol_note": system.protocol_note}
    job.update(rec)
    heartbeat.tick(0, system=key, stage="orbitals",
                   converged=int(rec["converged"]), cpu_s=rec["cpu_s"])
    record.add_job(job)
    return job


#: Sweep convergence for a ladder point. ⚠ **Deliberately looser than the solver's own
#: default, and it is a statement about what is being measured.** This protocol measures
#: TRUNCATION error against an exact CI: at a binding cap that error is 1e-2 to 1e-1 Eh,
#: so a sweep whose energy has stopped moving at 1e-8 Eh has converged for this purpose by
#: six or seven orders of magnitude. Holding it to the 1e-9 default instead spends a full
#: solve and then discards the point as unconverged — measured on DyCl3 at D = 6, which
#: stopped at 4.6e-8 Eh after the whole 30-sweep budget. The sweep count and the achieved
#: convergence are recorded per point either way, so a point that ran up against the budget
#: is visible rather than inferred.
LADDER_CONV_TOL = 1.0e-8


def stage_ladder(key: str, topologies: Sequence[str], record: Record, heartbeat, *,
                 deadline: float, caps: Optional[Sequence[int]] = None,
                 max_sweeps: int = 30) -> Dict:
    """One system's bond-dimension ladder on every stated topology.

    The oracle is solved once and reused for every point: it depends on the integrals, and
    the integrals do not move inside a ladder.
    """
    from kuiva.util import resources as res

    system = camp.get(key)
    res.clear()
    t_job = time.time()
    reference = camp.build_reference(system)
    if system.orbitals == "guess":
        from kuiva.interface import api
        space = api.active_space_for(reference, **system.selection())
        coeff = np.ascontiguousarray(reference.spinors_in_ao())
        orb = {"source": "scalar guess", "converged": True,
               "note": system.protocol_note}
    else:
        coeff, space, orb = camp.converged_orbitals(
            system, reference, deadline=max(0.0, deadline - time.time() - 120.0))
        orb["source"] = "SA-CASSCF"
    if not orb["converged"]:
        raise RuntimeError(
            "{}: the reference CASSCF is not converged (|grad| = {:.2e}); the ladder "
            "measures truncation against an exact CI at THESE orbitals, and an "
            "unconverged set makes the comparison a statement about the optimizer "
            "instead. Re-run the orbitals stage — it resumes from its checkpoint"
            .format(key, orb["grad_norm"]))
    ints = camp.cas_integrals(reference, coeff, space)

    e_ci, tdm_ci, ci_cost = camp.exact_ci(ints, system)
    m_ci = camp.property_matrices(reference, coeff, space, tdm_ci, e_ci, system)
    blocks = camp.oracle_blocks(m_ci, system)
    ref_reduction = camp.reduce_at(m_ci, blocks)
    e_sa_ci = float(np.mean(e_ci))

    graphs, topo_meta = camp.topologies(ints, system, topologies)
    job = {"key": key, "stage_kind": "ladder", "label": system.label,
           "roots": system.roots, "n_active": system.n_active,
           "n_active_elec": system.n_active_elec, "n_det": ci_cost["ndet"],
           "active_space": space.description, "orbitals": orb,
           "protocol_note": system.protocol_note,
           "sweep_budget": max_sweeps, "conv_tol": LADDER_CONV_TOL,
           "oracle": {"cost": ci_cost, "e_sa_eh": e_sa_ci,
                      "reduction": ref_reduction},
           "partition": topo_meta,
           "topologies": {name: {"edges": [list(map(int, e)) for e in g.edges],
                                 "contents": [list(map(int, c)) for c in g.contents]}
                          for name, g in graphs.items()}}
    if not record.has_job(key):
        record.add_job(job)
    print("  [{}] oracle: {} dets, {} roots, {} blocks, {:.1f} CPU s".format(
        key, ci_cost["ndet"], system.roots, len(blocks), ci_cost["cpu_s"]), flush=True)
    print("  [{}] nodes {} (two-site floor {} vs {} roots)".format(
        key, topo_meta["node_sizes"], topo_meta["two_site_floor"], system.roots),
        flush=True)

    ladder_caps = list(system.caps if caps is None else caps)
    already = record.done_points()
    n_point = 0
    for name in topologies:
        graph = graphs[name]
        for cap in ladder_caps:
            if (key, name, int(cap)) in already:
                print("  [{}/{}] D={:4d}  already measured; kept".format(key, name, cap),
                      flush=True)
                continue
            if time.time() > deadline:
                point = {"key": key, "topology": name, "cap": int(cap),
                         "status": "skipped", "reason": "stage wall budget exhausted"}
                record.add_point(point)
                print("  [{}/{}] budget exhausted at D={}".format(key, name, cap),
                      flush=True)
                break
            point = _ladder_point(system, reference, coeff, space, ints, graph, name, cap,
                                  blocks, ref_reduction, e_sa_ci, max_sweeps)
            point["elapsed_s"] = round(time.time() - t_job, 1)
            record.add_point(point)
            n_point += 1
            heartbeat.tick(n_point, system=key, topology=name, cap=int(cap),
                           status=point["status"], grade=point.get("grade", {}).get(
                               "overall", "-"))
            _print_point(key, name, cap, point)
            # ⚠ The ladder stops at saturation, and that is not a budget compromise: once
            # the state's own bond dimension sits below the cap the cap is not binding, so
            # every higher rung solves the SAME variational problem at more cost. The
            # already-validated saturating regime is exactly what this campaign is not
            # about.
            if point["status"] == "ok" and point["saturating"] \
                    and point["grade"]["overall"] == "quantitative":
                record.add_point({"key": key, "topology": name, "cap": None,
                                  "status": "ladder-complete",
                                  "reason": "saturated at D = {} (bond used {}), and the "
                                            "cap is no longer binding".format(
                                                cap, point["bond_used"])})
                print("  [{}/{}] saturated at D={} (bond used {}); higher caps solve the "
                      "same problem".format(key, name, cap, point["bond_used"]), flush=True)
                break
    return job


def _ladder_point(system, reference, coeff, space, ints, graph, topology: str, cap: int,
                  blocks, ref_reduction, e_sa_ci: float, max_sweeps: int) -> Dict:
    from kuiva.props.multiplet import HARTREE_TO_CM

    e_net, tdm_net, meta = camp.network_ci(ints, system, max_bond=cap, graph=graph,
                                           max_sweeps=max_sweeps,
                                           conv_tol=LADDER_CONV_TOL)
    point: Dict = {"key": system.key, "topology": topology, "cap": int(cap)}
    point.update(meta)
    if e_net is None:
        return point
    m_net = camp.property_matrices(reference, coeff, space, tdm_net, e_net, system)
    trial = camp.reduce_at(m_net, blocks)
    e_sa_error_eh = float(np.mean(e_net)) - e_sa_ci
    e_sa_error_cm = e_sa_error_eh * HARTREE_TO_CM
    point["e_sa_error_eh"] = e_sa_error_eh
    point["e_sa_error_cm"] = round(e_sa_error_cm, 6)
    point["reduction"] = trial
    point["grade"] = camp.grade(ref_reduction, trial, e_sa_error_cm=e_sa_error_cm)
    return point


def _print_point(key: str, topology: str, cap: int, point: Dict) -> None:
    if point["status"] != "ok":
        print("  [{}/{}] D={:4d}  {:11s}  {}".format(
            key, topology, cap, point["status"].upper(),
            point.get("error", "")[:96]), flush=True)
        return
    g = point["grade"]
    print("  [{}/{}] D={:4d}  used {:3d}  {:2d} sweeps  w_disc {:.2e}  "
          "dE_SA {:+.3e} Eh  dE_max {:8.3f} cm^-1  dg {:.2e}  {:14s}  {:.1f} CPU s".format(
              key, topology, cap, point["bond_used"], point["n_sweeps"], point["w_disc"],
              point["e_sa_error_eh"],
              g["max_energy_dev_cm"] if g["max_energy_dev_cm"] is not None else float("nan"),
              g["max_g_rel_dev"] if g["max_g_rel_dev"] is not None else float("nan"),
              g["overall"], point["cpu_s"]), flush=True)


def regrade(stages: Sequence[str]) -> Dict:
    """Re-apply the grading to every stored point, from its stored reduction.

    ⚠ **This is why the raw reduction is written into every record and not just the
    verdict.** A tier is a judgement laid over measurements, and a judgement can be found
    wrong — the g metric's absolute floor was added after the transition-metal ladders had
    already run, because a block with no moment was being graded against itself. Re-running
    the physics to change a band would be both wasteful and a temptation to leave the band
    where it is. Nothing here recomputes; the energies, moments and costs are untouched.
    """
    changed: Dict[str, int] = {}
    for stage in stages:
        path = camp.RECORDS / "{}.json".format(stage)
        if not path.is_file():
            continue
        data = json.loads(path.read_text())
        refs = {j["key"]: j["oracle"]["reduction"] for j in data.get("jobs", [])
                if j.get("stage_kind") == "ladder"}
        n = 0
        for point in data.get("points", []):
            if point.get("status") != "ok" or "reduction" not in point:
                continue
            ref = refs.get(point["key"])
            if ref is None:
                continue
            before = point.get("grade", {}).get("overall")
            point["grade"] = camp.grade(ref, point["reduction"],
                                        e_sa_error_cm=point.get("e_sa_error_cm"))
            if point["grade"]["overall"] != before:
                n += 1
        data["bands"] = camp.BANDS.__dict__
        data["regraded"] = True
        with open(path, "w") as fh:
            json.dump(data, fh, indent=1, sort_keys=True, default=_jsonable)
        changed[stage] = n
    return changed


# --- the deliverable: the cheapest settings reaching each tier -------------------------------
def summarize(stages: Sequence[str]) -> Dict:
    """Per (system, topology): the smallest cap reaching each tier, and what it cost.

    ⚠ **Two numbers, not one.** The quantitative floor says where the truncated result
    becomes interchangeable with the exact one; the qualitative floor says where it becomes
    a fast calculation worth running, which is the answer to "how little cost suffices".
    Where a ladder passes straight from quantitative to unacceptable with no usable
    qualitative window — plausible near free-ion degeneracy, where the group-complete
    truncation rule refuses the intermediate caps — that absence is reported explicitly
    rather than left as a blank.

    The cost figures are **CPU seconds**, always: this machine throttles under sustained
    load, so wall time here is partly a temperature measurement and is recorded but never
    judged.
    """
    rows: Dict[str, Dict] = {}
    for stage in stages:
        path = camp.RECORDS / "{}.json".format(stage)
        if not path.is_file():
            continue
        data = json.loads(path.read_text())
        oracle = {j["key"]: j for j in data.get("jobs", [])
                  if j.get("stage_kind") == "ladder"}
        for point in data.get("points", []):
            if point.get("cap") is None:
                continue          # the ladder-complete / skipped sentinels carry no cap
            key = point["key"]
            lane = "{}/{}".format(key, point["topology"])
            row = rows.setdefault(lane, {
                "key": key, "topology": point["topology"], "stage": stage,
                "roots": oracle.get(key, {}).get("roots"),
                "n_det": oracle.get(key, {}).get("n_det"),
                "oracle_cpu_s": oracle.get(key, {}).get("oracle", {})
                                      .get("cost", {}).get("cpu_s"),
                "partition": oracle.get(key, {}).get("partition", {}).get("node_sizes"),
                "caps_run": [], "refused": [], "unconverged": [],
                "floors": {}, "curve": []})
            cap = int(point["cap"])
            if point.get("status") == "refused":
                row["refused"].append(cap)
                continue
            if point.get("status") == "unconverged":
                row["unconverged"].append(cap)
                continue
            if point.get("status") != "ok":
                continue
            row["caps_run"].append(cap)
            grade = point["grade"]
            row["curve"].append({
                "cap": cap, "bond_used": point["bond_used"],
                "sweeps": point["n_sweeps"], "w_disc": point["w_disc"],
                "cpu_s": point["cpu_s"], "grade": grade["overall"],
                "max_energy_dev_cm": grade["max_energy_dev_cm"],
                "max_g_rel_dev": grade["max_g_rel_dev"],
                "e_sa_error_eh": point["e_sa_error_eh"]})
    for row in rows.values():
        row["curve"].sort(key=lambda c: c["cap"])
        for tier in ("quantitative", "qualitative"):
            reached = [c for c in row["curve"]
                       if camp.TIERS.index(c["grade"]) <= camp.TIERS.index(tier)]
            if reached:
                best = reached[0]
                row["floors"][tier] = {
                    "cap": best["cap"], "bond_used": best["bond_used"],
                    "sweeps": best["sweeps"], "cpu_s": best["cpu_s"],
                    "w_disc": best["w_disc"],
                    "cpu_ratio_to_exact_ci": (
                        None if not row["oracle_cpu_s"]
                        else round(best["cpu_s"] / row["oracle_cpu_s"], 2))}
            else:
                row["floors"][tier] = None
        # ⚠ Stated, not left blank: a ladder with no usable qualitative window is a finding.
        q, qq = row["floors"]["quantitative"], row["floors"]["qualitative"]
        row["qualitative_window"] = (
            "none reached" if qq is None else
            ("absent: the cheapest qualitative point is already the quantitative one"
             if q is not None and q["cap"] == qq["cap"] else "present"))
    return rows


def print_summary(rows: Dict) -> None:
    head = ("{:<16s} {:>5s} {:>6s} {:>11s} {:>11s} {:>9s} {:>9s}  {}"
            .format("system/topology", "roots", "dets", "quant D", "qual D",
                    "quant CPU", "qual CPU", "refused caps"))
    print("\n" + head)
    print("-" * len(head))
    for lane in sorted(rows):
        r = rows[lane]
        q, qq = r["floors"]["quantitative"], r["floors"]["qualitative"]
        print("{:<16s} {:>5s} {:>6s} {:>11s} {:>11s} {:>9s} {:>9s}  {}".format(
            lane, str(r["roots"] or "-"), str(r["n_det"] or "-"),
            str(q["cap"]) if q else "-", str(qq["cap"]) if qq else "-",
            "{:.1f}".format(q["cpu_s"]) if q else "-",
            "{:.1f}".format(qq["cpu_s"]) if qq else "-",
            ",".join(str(c) for c in sorted(r["refused"])) or "-"))


# --- driver ---------------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True,
                    choices=sorted(STAGES) + ["summary", "regrade"])
    ap.add_argument("--only", default=None, help="restrict to one system key")
    ap.add_argument("--budget", type=float, default=None,
                    help="hard wall budget for the stage [s]; the stage's own default "
                         "otherwise")
    ap.add_argument("--caps", default=None,
                    help="comma-separated bond caps, overriding the system's ladder")
    ap.add_argument("--max-sweeps", type=int, default=30)
    ap.add_argument("--fresh", action="store_true",
                    help="discard this stage's existing record instead of adding to it")
    args = ap.parse_args(argv)

    if args.stage == "regrade":
        changed = regrade(sorted(k for k in STAGES if k.startswith("s")))
        for stage, n in sorted(changed.items()):
            print("{}: {} point(s) changed tier".format(stage, n))
        return 0

    if args.stage == "summary":
        rows = summarize(sorted(k for k in STAGES if k.startswith("s1")))
        camp.RECORDS.mkdir(parents=True, exist_ok=True)
        path = camp.RECORDS / "summary.json"
        with open(path, "w") as fh:
            json.dump({"schema": SCHEMA, "bands": camp.BANDS.__dict__, "lanes": rows},
                      fh, indent=1, sort_keys=True, default=_jsonable)
        print_summary(rows)
        print("\n-> {}".format(path))
        return 0

    plan = STAGES[args.stage]
    budget = float(plan["budget"] if args.budget is None else args.budget)
    caps = None if args.caps is None else [int(x) for x in args.caps.split(",")]

    log_path = _setup_logging(args.stage)
    meta = _env_meta()
    meta["label"] = plan["label"]
    meta["budget_s"] = budget
    meta["log"] = str(log_path)
    record = Record(args.stage, camp.RECORDS / "{}.json".format(args.stage), meta,
                    resume=not args.fresh)

    jobs = [(k, t) for k, t in plan["jobs"] if args.only is None or k == args.only]
    heartbeat = Heartbeat("dmrg_cost_ladder_{}".format(args.stage), budget_seconds=budget,
                          meta={"stage": args.stage, "jobs": [k for k, _ in jobs]})
    t_start = time.time()
    deadline = t_start + budget
    print("stage {} ({}): {} job(s), budget {:.0f} s, log {}".format(
        args.stage, plan["label"], len(jobs), budget, log_path), flush=True)

    status = 0
    for key, topologies in jobs:
        if time.time() > deadline:
            record.data.setdefault("stopped_early", []).append(
                "{}: stage wall budget exhausted before it started".format(key))
            record.flush()
            print("budget exhausted before {}".format(key), flush=True)
            break
        try:
            if plan["kind"] == "front-end":
                stage_front_end(key, record, heartbeat)
            elif plan["kind"] == "orbitals":
                stage_orbitals(key, record, heartbeat, deadline=deadline)
            else:
                stage_ladder(key, topologies, record, heartbeat, deadline=deadline,
                             caps=caps if caps is not None else plan.get("caps"),
                             max_sweeps=args.max_sweeps)
        except Exception as exc:                     # a failed job is data, not a crash
            import traceback
            record.data.setdefault("failures", []).append(
                {"key": key, "error": "{}: {}".format(type(exc).__name__, exc),
                 "traceback": traceback.format_exc()})
            record.flush()
            print("  [{}] FAILED: {}: {}".format(key, type(exc).__name__, exc), flush=True)
            status = 1
    record.data["elapsed_s"] = round(time.time() - t_start, 1)
    record.flush()
    heartbeat.finish(elapsed=time.time() - t_start, status=status)
    print("\nstage {} done in {:.0f} s -> {}".format(args.stage, time.time() - t_start,
                                                     record.path), flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
