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
    # --- protocol B: the truncation with orbital feedback, not at fixed orbitals ---------
    # ⚠ Caps are chosen FROM protocol A's measured curve, one per tier where the tier
    # exists, cheapest system first so the structural answer arrives before the expensive
    # one is attempted. `dycl3` gets its cheapest reachable cap only: at 1 842 CPU s per
    # network solve, a full optimization of it is a multi-day run and the plan's
    # "30 min - 1.5 h per cap" estimate predates s1.2's measurement of what a point costs.
    "s1.5": dict(kind="relax", label="protocol B: DMRG-CASSCF at truncating caps",
                 jobs=(("tif3", ("path",)), ("fecl2", ("path",)), ("dycl3", ("path",))),
                 relax_caps={"tif3": (2, 4, 6), "fecl2": (4, 16, 24, 32), "dycl3": (4,)},
                 relax_budget={"tif3": 600.0, "fecl2": 1800.0, "dycl3": 3000.0},
                 relax_max_iter={"tif3": 60, "fecl2": 60, "dycl3": 30},
                 budget=3.0 * 3600),
    # ⚠ The remedy leg, and it is a measurement rather than a fix: the smooth-surface
    # driver stalls at a binding cap with its trust radius collapsing and its gradient
    # frozen, which is the signature of an energy that is not a function of the rotation
    # alone. The event-gated driver is the one designed for that surface, and whether it
    # recovers anything at a truncating cap is the open question this leg answers.
    "s1.5c": dict(kind="relax", label="protocol B with the event-gated driver",
                  jobs=(("fecl2", ("path",)),),
                  relax_caps={"fecl2": (4, 24)},
                  relax_budget={"fecl2": 1800.0},
                  relax_max_iter={"fecl2": 60},
                  relax_drivers=("events",),
                  budget=1.5 * 3600),
    "s1.6": dict(kind="ladder", label="beyond-minimal active spaces with an exact oracle",
                 jobs=(("tif3_dd", ("path", "tree")), ("fecl2_dd", ("path", "tree"))),
                 budget=3.0 * 3600),
    "s1.7": dict(kind="ladder", label="DyCl3 rung A: 4f + the Cl sigma-donor set (three pairs)",
                 jobs=(("dycl3_a", ("path",)),), budget=3.0 * 3600),
    "s1.8": dict(kind="ladder", label="DyCl3 rung B: 4f + 5d",
                 jobs=(("dycl3_b", ("path",)),), budget=3.0 * 3600),
    # ⚠ Protocol C runs on the system with the WIDEST useful ladder, which the measurements
    # pick rather than the plan: `fecl2` is the only system in the campaign with a
    # qualitative window at all (D = 24) and a quantitative point above it (D = 32).
    "s1.9": dict(kind="controls", label="production controls: ramps and subspace expansion",
                 jobs=(("fecl2", ("path",)),),
                 control_caps={"fecl2": (4, 8, 16, 24)},
                 control_variants=("ramp", "expand", "ramp+expand"),
                 budget=2.0 * 3600),
    "s1.6b": dict(kind="relax", label="protocol B on a beyond-minimal active space",
                  jobs=(("tif3_dd", ("path",)),),
                  relax_caps={"tif3_dd": (4, 8)},
                  relax_budget={"tif3_dd": 900.0},
                  relax_max_iter={"tif3_dd": 60},
                  budget=1.0 * 3600),
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
    camp.clear_templates()
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

    # ⚠ **The oracle may not exist, and its absence is a planned branch rather than a
    # failure.** Past the conventional-CI ceiling the memory pre-flight refuses before it
    # allocates — the honest verdict, and it costs nothing — and the ladder then has no
    # exact answer to be graded against. It is run anyway, in the mode Tier 3 will have to
    # use: reduced at the manifold structure the rungs below establish, and graded for
    # **internal convergence in D** against its own largest cap.
    oracle_refusal = None
    try:
        if not system.exact_oracle:
            # ⚠ Declared absent, not skipped for convenience: the pre-flight refused this
            # oracle on this machine (the refusal is in the stage record that measured it),
            # and attempting it under an overcommit would trade a diagnosed refusal for an
            # OOM kill — which is exactly the trade the memory pre-flight exists to
            # prevent.
            raise MemoryError(
                "the exact CI oracle for {} is declared unavailable: the memory pre-flight "
                "refused it on this machine, so the ladder runs oracle-free"
                .format(system.key))
        e_ci, tdm_ci, ci_cost = camp.exact_ci(ints, system)
    except MemoryError as exc:
        oracle_refusal = "{}: {}".format(type(exc).__name__, exc)
        e_ci = tdm_ci = None
        ci_cost = {"ndet": int(system.n_det), "wall_s": 0.0, "cpu_s": 0.0,
                   "status": "refused", "reason": oracle_refusal}
    if e_ci is None:
        # Consecutive Kramers doublets: the physical manifold statement the rungs below
        # measured, never an inference from a truncated spectrum.
        blocks = [(2 * i, 2) for i in range(system.roots // 2)]
        ref_reduction = None
        e_sa_ci = None
        print("  [{}] ORACLE REFUSED: {}".format(key, oracle_refusal[:160]), flush=True)
    else:
        m_ci = camp.property_matrices(reference, coeff, space, tdm_ci, e_ci, system)
        blocks = camp.oracle_blocks(m_ci, system)
        ref_reduction = camp.reduce_at(m_ci, blocks)
        e_sa_ci = float(np.mean(e_ci))

    graphs, topo_meta = camp.topologies(ints, system, topologies)
    # compile every topology's operator once, up front, and record what it cost: the
    # points below reuse it and their CPU figures are the solve alone
    compile_cost = {name: camp.compiled_template(g)[1] for name, g in graphs.items()}
    job = {"key": key, "stage_kind": "ladder", "label": system.label,
           "roots": system.roots, "n_active": system.n_active,
           "n_active_elec": system.n_active_elec, "n_det": ci_cost["ndet"],
           "active_space": space.description, "orbitals": orb,
           "protocol_note": system.protocol_note,
           "sweep_budget": max_sweeps, "conv_tol": LADDER_CONV_TOL,
           "oracle": {"cost": ci_cost, "e_sa_eh": e_sa_ci,
                      "reduction": ref_reduction, "refused": oracle_refusal,
                      "blocks": [[int(a), int(b)] for a, b in blocks]},
           "partition": topo_meta,
           "topologies": {name: {"edges": [list(map(int, e)) for e in g.edges],
                                 "contents": [list(map(int, c)) for c in g.contents],
                                 **compile_cost[name]}
                          for name, g in graphs.items()}}
    if not record.has_job(key):
        record.add_job(job)
    if ref_reduction is not None:
        print("  [{}] oracle: {} dets, {} roots, {} blocks, {:.1f} CPU s".format(
            key, ci_cost["ndet"], system.roots, len(blocks), ci_cost["cpu_s"]),
            flush=True)
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
            # ⚠ an oracle-free point has no grade (it is filled in afterwards against the
            # ladder's own largest cap), so the saturation test reads it defensively
            if point["status"] == "ok" and point["saturating"] \
                    and point.get("grade", {}).get("overall") == "quantitative":
                record.add_point({"key": key, "topology": name, "cap": None,
                                  "status": "ladder-complete",
                                  "reason": "saturated at D = {} (bond used {}), and the "
                                            "cap is no longer binding".format(
                                                cap, point["bond_used"])})
                print("  [{}/{}] saturated at D={} (bond used {}); higher caps solve the "
                      "same problem".format(key, name, cap, point["bond_used"]), flush=True)
                break
        if ref_reduction is None:
            internal = _internal_grade(record, key, name)
            if internal is not None:
                record.data.setdefault("internal_convergence", []).append(internal)
                record.flush()
                print("  [{}/{}] internal convergence against D={}: {}".format(
                    key, name, internal["reference_cap"],
                    ", ".join("D{}={}".format(r["cap"], r["grade"]["overall"])
                              for r in internal["rows"])), flush=True)
    return job


def _ladder_point(system, reference, coeff, space, ints, graph, topology: str, cap: int,
                  blocks, ref_reduction, e_sa_ci: Optional[float],
                  max_sweeps: int) -> Dict:
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
    point["reduction"] = trial
    point["e_sa_eh"] = float(np.mean(e_net))
    # ⚠ Without an oracle there is no error to quote and none is invented: the point carries
    # its reduction and its own state-averaged energy, and the tier it would have been given
    # is filled in afterwards, against the ladder's own largest cap (:func:`_internal_grade`).
    if ref_reduction is None:
        point["graded"] = "deferred: no oracle, internal convergence only"
        return point
    e_sa_error_eh = point["e_sa_eh"] - e_sa_ci
    e_sa_error_cm = e_sa_error_eh * HARTREE_TO_CM
    point["e_sa_error_eh"] = e_sa_error_eh
    point["e_sa_error_cm"] = round(e_sa_error_cm, 6)
    point["grade"] = camp.grade(ref_reduction, trial, e_sa_error_cm=e_sa_error_cm)
    return point


def _internal_grade(record: Record, key: str, topology: str) -> Optional[Dict]:
    """Grade an oracle-free ladder against **its own largest cap** — convergence in D.

    ⚠ **This is not a claim of accuracy and the record says so in the field name.** It
    answers the only question available past the CI ceiling: has the observable stopped
    moving as the bond dimension grows? A ladder that has converged internally may still be
    converged to the wrong thing — the tensor-network literature's standard caution, and the
    reason the rungs below it, which *do* have an oracle, are what license the method here.
    """
    points = [p for p in record.data["points"]
              if p.get("key") == key and p.get("topology") == topology
              and p.get("status") == "ok" and "reduction" in p]
    if len(points) < 2:
        return None
    points.sort(key=lambda p: int(p["cap"]))
    top = points[-1]
    rows = []
    for p in points[:-1]:
        from kuiva.props.multiplet import HARTREE_TO_CM
        de_cm = (float(p["e_sa_eh"]) - float(top["e_sa_eh"])) * HARTREE_TO_CM
        grade = camp.grade(top["reduction"], p["reduction"], e_sa_error_cm=de_cm)
        rows.append({"cap": int(p["cap"]), "against_cap": int(top["cap"]),
                     "e_sa_diff_cm": round(de_cm, 6), "grade": grade,
                     "w_disc": p.get("w_disc"), "cpu_s": p.get("cpu_s")})
        p["internal_grade"] = grade["overall"]
    return {"key": key, "topology": topology, "reference_cap": int(top["cap"]),
            "reference_w_disc": top.get("w_disc"), "rows": rows,
            "note": "convergence in D against the ladder's own largest cap; not accuracy"}


def _print_point(key: str, topology: str, cap: int, point: Dict) -> None:
    if point["status"] != "ok":
        print("  [{}/{}] D={:4d}  {:11s}  {}".format(
            key, topology, cap, point["status"].upper(),
            point.get("error", "")[:96]), flush=True)
        return
    g = point.get("grade")
    if g is None:                              # oracle-free: graded afterwards, internally
        print("  [{}/{}] D={:4d}  used {:3d}  {:2d} sweeps  w_disc {:.2e}  E_SA {:.8f} Eh  "
              "{:14s}  {:.1f} CPU s".format(
                  key, topology, cap, point["bond_used"], point["n_sweeps"],
                  point["w_disc"], point["e_sa_eh"], "oracle-free", point["cpu_s"]),
              flush=True)
        return
    print("  [{}/{}] D={:4d}  used {:3d}  {:2d} sweeps  w_disc {:.2e}  "
          "dE_SA {:+.3e} Eh  dE_max {:8.3f} cm^-1  dg {:.2e}  {:14s}  {:.1f} CPU s".format(
              key, topology, cap, point["bond_used"], point["n_sweeps"], point["w_disc"],
              point["e_sa_error_eh"],
              g["max_energy_dev_cm"] if g["max_energy_dev_cm"] is not None else float("nan"),
              g["max_g_rel_dev"] if g["max_g_rel_dev"] is not None else float("nan"),
              g["overall"], point["cpu_s"]), flush=True)


# --- protocol B: does orbital relaxation absorb the truncation error, or amplify it? --------
def _protocol_a_point(key: str, topology: str, cap: int) -> Optional[Dict]:
    """The fixed-orbital ladder point for the same (system, topology, cap), from its record.

    ⚠ Read rather than recomputed, and it is the *same* orbitals: protocol A ran at the
    converged CI-CASSCF orbitals this stage takes from the same checkpoint. Recomputing it
    here would cost a full network solve to reproduce a number already measured, and a
    second measurement of the same quantity is how two slightly different protocols end up
    being compared with each other.
    """
    for stage in sorted(k for k in STAGES if k.startswith("s")):
        path = camp.RECORDS / "{}.json".format(stage)
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text())
        except ValueError:
            continue
        for point in data.get("points", []):
            if point.get("protocol") == "B":
                continue
            if (point.get("key") == key and point.get("topology") == topology
                    and point.get("cap") == int(cap)):
                return {"stage": stage, "status": point.get("status"),
                        "grade": point.get("grade", {}).get("overall"),
                        "max_energy_dev_cm": point.get("grade", {}).get(
                            "max_energy_dev_cm"),
                        "max_g_rel_dev": point.get("grade", {}).get("max_g_rel_dev"),
                        "e_sa_error_eh": point.get("e_sa_error_eh"),
                        "cpu_s": point.get("cpu_s")}
    return None


def stage_relax(key: str, topologies: Sequence[str], record: Record, heartbeat, *,
                deadline: float, caps: Sequence[int], wall_budget: float, max_iter: int,
                max_sweeps: int = 30,
                drivers: Sequence[str] = ("trust-region",)) -> Dict:
    """One system's protocol-B leg: a full DMRG-CASSCF at each stated cap.

    Both legs start from the **same** scalar-guess orbitals, use the same step engine and
    the same convergence criterion, and differ only in the CI solver. The CI leg is the
    campaign's own checkpointed reference CASSCF — which is exactly the orbital set protocol
    A graded against, so the two protocols are commensurable by construction rather than by
    a second optimization that would have to be argued to be the same one.

    ``drivers`` selects the outer control: ``"trust-region"`` is
    :func:`~kuiva.mcscf.orbopt.optimize_orbitals`, the validated smooth-surface driver;
    ``"events"`` is :func:`~kuiva.mcscf.events.optimize_orbitals_events`, whose whole reason
    for existing is a solver whose internal space moves with the orbitals. ⚠ A DMRG at a
    *binding* cap is exactly that even with a fixed graph and a fixed cap — the sweep is
    warm-started and only locally optimal inside its manifold, so the energy it returns is
    not a function of the rotation alone — which is why the second driver is a leg of this
    protocol rather than a footnote to it.

    ⚠ **The state-averaging gate is ON in this protocol** (``on_split="raise"``, the
    solver's default): here the averaged density really does move the orbitals, which is the
    situation the gate exists for. Its refusal at a truncating cap is therefore a
    measurement — "no orbital optimization may be built on this truncated ensemble" — and it
    is recorded as an outcome with the cap that produced it, never worked around.
    """
    from kuiva.util import resources as res

    system = camp.get(key)
    res.clear()
    camp.clear_templates()
    t_job = time.time()
    reference = camp.build_reference(system)
    coeff_ci, space, orb = camp.converged_orbitals(
        system, reference, deadline=max(0.0, deadline - time.time() - 120.0))
    if not orb["converged"]:
        raise RuntimeError(
            "{}: the CI-CASSCF leg is not converged (|grad| = {:.2e}); protocol B compares "
            "two converged optimizations and an unconverged reference makes every "
            "difference a statement about the optimizer".format(key, orb["grad_norm"]))
    ints_ci = camp.cas_integrals(reference, coeff_ci, space)
    e_ci, tdm_ci, ci_cost = camp.exact_ci(ints_ci, system)
    m_ci = camp.property_matrices(reference, coeff_ci, space, tdm_ci, e_ci, system)
    blocks = camp.oracle_blocks(m_ci, system)
    ref_reduction = camp.reduce_at(m_ci, blocks)
    e_sa_ci = float(np.mean(e_ci))

    # ⚠ The topology is stated on the STARTING orbitals, before either optimization runs,
    # and held fixed through both — the production choice, and the only one that keeps the
    # two legs comparable: a topology re-derived at each leg's own orbitals would make the
    # network a different approximation on each side of the comparison.
    c0 = np.ascontiguousarray(reference.spinors_in_ao())
    ints_0 = camp.cas_integrals(reference, c0, space)
    graphs, topo_meta = camp.topologies(ints_0, system, topologies)

    job = {"key": key, "stage_kind": "relax", "protocol": "B", "label": system.label,
           "roots": system.roots, "n_active": system.n_active,
           "n_active_elec": system.n_active_elec, "n_det": ci_cost["ndet"],
           "active_space": space.description, "protocol_note": system.protocol_note,
           "casscf_mode": system.casscf_mode, "max_iter": max_iter,
           "conv_grad": system.conv_grad, "sweep_budget": max_sweeps,
           "wall_budget_s": wall_budget,
           "ci_leg": dict(orb, source="CI-CASSCF (the campaign reference)",
                          e_sa_eh=e_sa_ci, oracle_cost=ci_cost,
                          reduction=ref_reduction),
           "partition": topo_meta}
    if not record.has_job(key):
        record.add_job(job)
    print("  [{}] CI-CASSCF: E = {:.8f} Eh, {} iterations, {:.1f} CPU s; oracle {} dets, "
          "{} blocks".format(key, orb["e_avg"], orb["iterations"], orb["cpu_s"],
                             ci_cost["ndet"], len(blocks)), flush=True)

    already = record.done_points()
    n_point = 0
    for name in topologies:
        graph = graphs[name]
        for driver in drivers:
            # ⚠ The lane, not just the point, carries the driver: the two are different
            # calculations of the same quantity and must never overwrite one another in a
            # resumed record.
            lane = name if driver == "trust-region" else "{}-{}".format(name, driver)
            for cap in caps:
                if (key, lane, int(cap)) in already:
                    print("  [{}/{}] D={:4d}  already measured; kept".format(
                        key, lane, cap), flush=True)
                    continue
                if time.time() > deadline:
                    record.add_point({"key": key, "topology": lane, "cap": int(cap),
                                      "protocol": "B", "status": "skipped",
                                      "reason": "stage wall budget exhausted"})
                    print("  [{}/{}] budget exhausted at D={}".format(key, lane, cap),
                          flush=True)
                    break
                budget = min(float(wall_budget), max(60.0, deadline - time.time() - 60.0))
                point = _relax_point(system, reference, space, graph, name, cap, c0,
                                     blocks, ref_reduction, e_sa_ci, orb,
                                     wall_budget=budget, max_iter=max_iter,
                                     max_sweeps=max_sweeps, driver=driver, lane=lane)
                point["elapsed_s"] = round(time.time() - t_job, 1)
                record.add_point(point)
                n_point += 1
                heartbeat.tick(n_point, system=key, topology=lane, cap=int(cap),
                               status=point["status"],
                               grade=point.get("grade", {}).get("overall", "-"))
                _print_relax_point(key, lane, cap, point)
    return job


def _relax_point(system, reference, space, graph, topology: str, cap: int,
                 c0: np.ndarray, blocks, ref_reduction, e_sa_ci: float, ci_leg: Dict, *,
                 wall_budget: float, max_iter: int, max_sweeps: int,
                 driver: str = "trust-region", lane: Optional[str] = None) -> Dict:
    """One (system, cap, driver): the DMRG-CASSCF, its final states, and both gradings."""
    from kuiva.props.multiplet import HARTREE_TO_CM

    point: Dict = {"key": system.key, "topology": lane or topology, "cap": int(cap),
                   "protocol": "B", "driver": driver,
                   "protocol_a": _protocol_a_point(system.key, topology, cap)}
    solver = camp.network_solver(system, max_bond=cap, graph=graph,
                                 max_sweeps=max_sweeps, conv_tol=LADDER_CONV_TOL)
    result, rec = camp.relax_orbitals(reference, c0, space, solver,
                                      mode=system.casscf_mode, max_iter=max_iter,
                                      conv_grad=system.conv_grad,
                                      wall_budget=wall_budget, report=True,
                                      driver=driver)
    point.update(rec)
    if result is None:
        return point
    # ⚠ The final states are re-solved at the relaxed orbitals through the ladder's own
    # analysis path (on_split="warn"), so the two protocols are reduced by exactly the same
    # code from exactly the same kind of object. The gate that matters was the one inside
    # the optimization, and it has already had its say.
    ints = camp.cas_integrals(reference, result.coeff, space)
    e_net, tdm_net, meta = camp.network_ci(ints, system, max_bond=cap, graph=graph,
                                           max_sweeps=max_sweeps,
                                           conv_tol=LADDER_CONV_TOL)
    point["final_solve"] = meta
    if e_net is None:
        point["status"] = "relaxed-but-{}".format(meta["status"])
        return point
    m_net = camp.property_matrices(reference, result.coeff, space, tdm_net, e_net, system)
    trial = camp.reduce_at(m_net, blocks)
    e_sa_error_eh = float(np.mean(e_net)) - e_sa_ci
    point["e_sa_error_eh"] = e_sa_error_eh
    point["e_sa_error_cm"] = round(e_sa_error_eh * HARTREE_TO_CM, 6)
    point["e_casscf_error_eh"] = float(rec["e_avg"]) - float(ci_leg["e_avg"])
    point["reduction"] = trial
    point["grade"] = camp.grade(ref_reduction, trial,
                                e_sa_error_cm=point["e_sa_error_cm"])
    point["cpu_ratio_to_ci_casscf"] = (
        None if not ci_leg.get("cpu_s") else round(rec["cpu_s"] / ci_leg["cpu_s"], 1))
    return point


def _print_relax_point(key: str, topology: str, cap: int, point: Dict) -> None:
    a = (point.get("protocol_a") or {}).get("grade", "-")
    if point["status"] != "ok":
        print("  [{}/{}] D={:4d}  {:20s}  (protocol A: {})  {}".format(
            key, topology, cap, point["status"].upper(), a,
            point.get("error", "")[:80]), flush=True)
        return
    g = point["grade"]
    print("  [{}/{}] D={:4d}  {:3d} iter  |g| {:.1e}  {:9s}  dE_CASSCF {:+.3e} Eh  "
          "dE_max {:9.3f} cm^-1  dg {:.2e}  B: {:14s}  A: {:14s}  {:.1f} CPU s".format(
              key, topology, cap, point["iterations"], point["grad_norm"],
              "converged" if point["converged"] else "NOT conv",
              point["e_casscf_error_eh"],
              g["max_energy_dev_cm"] if g["max_energy_dev_cm"] is not None else float("nan"),
              g["max_g_rel_dev"] if g["max_g_rel_dev"] is not None else float("nan"),
              g["overall"], str(a), point["cpu_s"]), flush=True)


# --- protocol C: the production controls, and whether extrapolation beats raising D ---------
def stage_controls(key: str, topologies: Sequence[str], record: Record, heartbeat, *,
                   deadline: float, caps: Sequence[int], variants: Sequence[str],
                   max_sweeps: int = 30) -> Dict:
    """Schedules and subspace expansion at a stated cap, against the plain solve there.

    ⚠ **Every variant is the same manifold**: a per-sweep bond ramp and a deterministic
    subspace expansion are *iteration strategies inside* the cap, not ways to exceed it, so
    the variational answer they converge to is the plain solve's. What they can change is
    which fixed point the sweeps find and how many sweeps that takes — which is exactly the
    question, because at a binding cap the sweeps do not always find the best state in the
    manifold they are allowed (the third gating mechanism: an intermediate cap that exhausts
    its sweep budget).

    The comparison is at **equal cap**, and cost is CPU seconds as everywhere here.
    """
    from kuiva.util import resources as res

    system = camp.get(key)
    res.clear()
    camp.clear_templates()
    t_job = time.time()
    reference = camp.build_reference(system)
    coeff, space, orb = camp.converged_orbitals(
        system, reference, deadline=max(0.0, deadline - time.time() - 120.0))
    if not orb["converged"]:
        raise RuntimeError("{}: the reference CASSCF is not converged".format(key))
    ints = camp.cas_integrals(reference, coeff, space)
    e_ci, tdm_ci, ci_cost = camp.exact_ci(ints, system)
    m_ci = camp.property_matrices(reference, coeff, space, tdm_ci, e_ci, system)
    blocks = camp.oracle_blocks(m_ci, system)
    ref_reduction = camp.reduce_at(m_ci, blocks)
    e_sa_ci = float(np.mean(e_ci))
    graphs, topo_meta = camp.topologies(ints, system, topologies)
    compile_cost = {name: camp.compiled_template(g)[1] for name, g in graphs.items()}

    job = {"key": key, "stage_kind": "controls", "label": system.label,
           "roots": system.roots, "n_active": system.n_active,
           "n_active_elec": system.n_active_elec, "n_det": ci_cost["ndet"],
           "active_space": space.description, "orbitals": orb,
           "protocol_note": system.protocol_note, "variants": list(variants),
           "oracle": {"cost": ci_cost, "e_sa_eh": e_sa_ci, "reduction": ref_reduction},
           "partition": topo_meta}
    if not record.has_job(key):
        record.add_job(job)
    print("  [{}] oracle: {} dets, {} roots, {} blocks, {:.1f} CPU s".format(
        key, ci_cost["ndet"], system.roots, len(blocks), ci_cost["cpu_s"]), flush=True)

    already = record.done_points()
    n_point = 0
    for name in topologies:
        graph = graphs[name]
        for variant in variants:
            lane = "{}+{}".format(name, variant)
            for cap in caps:
                if (key, lane, int(cap)) in already:
                    print("  [{}/{}] D={:4d}  already measured; kept".format(
                        key, lane, cap), flush=True)
                    continue
                if time.time() > deadline:
                    record.add_point({"key": key, "topology": lane, "cap": int(cap),
                                      "variant": variant, "status": "skipped",
                                      "reason": "stage wall budget exhausted"})
                    break
                point = _control_point(system, reference, coeff, space, ints, graph, lane,
                                       cap, variant, blocks, ref_reduction, e_sa_ci,
                                       max_sweeps, ramp_floor=min(int(c) for c in caps))
                point["elapsed_s"] = round(time.time() - t_job, 1)
                record.add_point(point)
                n_point += 1
                heartbeat.tick(n_point, system=key, topology=lane, cap=int(cap),
                               status=point["status"],
                               grade=point.get("grade", {}).get("overall", "-"))
                _print_point(key, lane, cap, point)
    return job


#: Protocol C's variants, each a setting of the solver at an unchanged cap.
#: ``ramp`` is the per-sweep bond schedule of the FIRST solve (2 -> 4 -> ... -> cap), which
#: is how a production run avoids converging into a poor fixed point of the full manifold;
#: ``expand`` is deterministic subspace expansion, which perturbs a two-site update so the
#: sweep can leave a stationary point the truncation put it in.
CONTROL_EXPANSION = 1.0e-3


def _ramp_rungs(cap: int, floor: int) -> List[int]:
    """The per-sweep bond ramp, starting no lower than the ensemble allows.

    ⚠ **A ramp's first rung is bounded from below by the ROOT COUNT, not chosen freely**,
    and that is a finding rather than a detail of this helper. A two-site update has to hold
    the whole averaged ensemble, and at the start of a ramp the state's own bond dimension —
    not the cap — is what bounds that space: starting a 25-root FeCl2 solve at D = 2 gives a
    two-site space of dimension 20 and the sweep refuses outright, at every cap. The floor is
    the lowest cap the plain ladder reached on that system, which is the same quantity
    measured rather than derived.
    """
    rungs = [d for d in (2, 4, 8, 16, 32, 64, 128) if int(floor) <= d < int(cap)]
    return rungs + [int(cap)]


def _control_variant_kwargs(variant: str, cap: int, floor: int) -> Dict:
    if variant == "plain":
        return {}
    if variant == "ramp":
        return {"bond_schedule": _ramp_rungs(cap, floor)}
    if variant == "expand":
        return {"expansion": CONTROL_EXPANSION}
    if variant == "ramp+expand":
        return {"bond_schedule": _ramp_rungs(cap, floor),
                "expansion": CONTROL_EXPANSION}
    raise ValueError("unknown control variant {!r}".format(variant))


def _control_point(system, reference, coeff, space, ints, graph, lane: str, cap: int,
                   variant: str, blocks, ref_reduction, e_sa_ci: float,
                   max_sweeps: int, ramp_floor: int = 4) -> Dict:
    from kuiva.props.multiplet import HARTREE_TO_CM

    kw = _control_variant_kwargs(variant, cap, ramp_floor)
    e_net, tdm_net, meta = camp.network_ci(ints, system, max_bond=cap, graph=graph,
                                           max_sweeps=max_sweeps,
                                           conv_tol=LADDER_CONV_TOL, **kw)
    point: Dict = {"key": system.key, "topology": lane, "cap": int(cap),
                   "variant": variant, "settings": {k: list(v) if isinstance(v, list)
                                                    else v for k, v in kw.items()}}
    point.update(meta)
    if e_net is None:
        return point
    m_net = camp.property_matrices(reference, coeff, space, tdm_net, e_net, system)
    trial = camp.reduce_at(m_net, blocks)
    e_sa_error_eh = float(np.mean(e_net)) - e_sa_ci
    point["e_sa_error_eh"] = e_sa_error_eh
    point["e_sa_error_cm"] = round(e_sa_error_eh * HARTREE_TO_CM, 6)
    point["reduction"] = trial
    point["grade"] = camp.grade(ref_reduction, trial, e_sa_error_cm=point["e_sa_error_cm"])
    return point


# --- protocol C, the free half: does extrapolating a cheap series beat one expensive run? ---
def extrapolate(stage: str, key: str, topology: str,
                caps: Optional[Sequence[int]] = None) -> Optional[Dict]:
    """Extrapolate a measured ladder to zero discarded weight and grade the result.

    ⚠ **No new physics is computed here, and that is the point.** Every ladder point already
    stores its discarded weight and its full phase-invariant reduction, so the standard
    ``E(w_disc -> 0)`` extrapolation is a straight line through numbers on disk — and the
    question protocol C asks is a *cost* question: can a series of cheap, qualitative-tier
    points, extrapolated, deliver a better tier than the cheapest single point that reaches
    it, for less CPU?

    The fit is linear in ``w_disc`` and is applied **per block** to the relative energies and
    to each principal g value, because those are the graded quantities; the reduction that
    comes out is graded by the ordinary :func:`dmrg_campaign.grade` against the same oracle.
    ⚠ A block whose reference carries no moment is left alone rather than extrapolated: its
    g values are noise about zero and a line through noise is noise with a slope.
    """
    path = camp.RECORDS / "{}.json".format(stage)
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    jobs = {j["key"]: j for j in data.get("jobs", [])
            if j.get("stage_kind") in ("ladder", "controls")}
    if key not in jobs:
        return None
    ref = jobs[key]["oracle"]["reduction"]
    points = [p for p in data.get("points", [])
              if p.get("key") == key and p.get("topology") == topology
              and p.get("status") == "ok" and p.get("cap") is not None
              and (caps is None or int(p["cap"]) in set(int(c) for c in caps))]
    points.sort(key=lambda p: int(p["cap"]))
    if len(points) < 2:
        return None
    w = np.array([float(p["w_disc"]) for p in points])
    if np.ptp(w) <= 0.0:
        return None                      # a series with one discarded weight fits nothing

    def fit(values: Sequence[float]) -> float:
        a = np.asarray(values, dtype=float)
        slope, intercept = np.polyfit(w, a, 1)
        del slope
        return float(intercept)

    n_block = min(len(p["reduction"]["blocks"]) for p in points)
    blocks: List[Dict] = []
    for k in range(n_block):
        rows = [p["reduction"]["blocks"][k] for p in points]
        g_ref = ref["blocks"][k]["g"] if k < len(ref["blocks"]) else None
        g_out = None
        if all(r["g"] is not None for r in rows) and g_ref is not None:
            n_g = min(len(r["g"]) for r in rows)
            if max(abs(x) for x in g_ref) < camp.BANDS.g_floor:
                g_out = [float(rows[-1]["g"][i]) for i in range(n_g)]
            else:
                g_out = [fit([r["g"][i] for r in rows]) for i in range(n_g)]
        blocks.append({"start": rows[-1]["start"], "size": rows[-1]["size"],
                       "energy_cm": fit([r["energy_cm"] for r in rows]),
                       "spread_cm": float(rows[-1]["spread_cm"]),
                       "g": g_out,
                       "character": (camp.axial_character(g_out) if g_out else None)})
    trial = {"levels_cm": points[-1]["reduction"]["levels_cm"], "blocks": blocks,
             "own_pattern": points[-1]["reduction"]["own_pattern"]}
    e_cm = fit([float(p["e_sa_error_cm"]) for p in points])
    grade = camp.grade(ref, trial, e_sa_error_cm=e_cm)
    cpu = float(sum(float(p["cpu_s"]) for p in points))
    return {"stage": stage, "key": key, "topology": topology,
            "caps": [int(p["cap"]) for p in points],
            "w_disc": [float(x) for x in w],
            "e_sa_error_cm_extrapolated": e_cm,
            "cpu_s_total": cpu,
            "oracle_cpu_s": jobs[key]["oracle"]["cost"]["cpu_s"],
            "grade": grade,
            "per_point_grade": [p["grade"]["overall"] for p in points]}


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
        # ⚠ A protocol-B point is graded against its own leg's CI-CASSCF reduction, not
        # against the fixed-orbital oracle: the two references are at different orbitals.
        refs.update({j["key"]: j["ci_leg"]["reduction"] for j in data.get("jobs", [])
                     if j.get("stage_kind") == "relax"})
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
            if point.get("protocol") == "B":
                continue          # a different protocol; :func:`summarize_relax` reads those
            key = point["key"]
            lane = "{}/{}".format(key, point["topology"])
            row = rows.setdefault(lane, {
                "key": key, "topology": point["topology"], "stage": stage,
                "roots": oracle.get(key, {}).get("roots"),
                "n_det": oracle.get(key, {}).get("n_det"),
                "oracle_cpu_s": oracle.get(key, {}).get("oracle", {})
                                      .get("cost", {}).get("cpu_s"),
                "partition": oracle.get(key, {}).get("partition", {}).get("node_sizes"),
                "caps_run": [], "refused": [], "refused_memory": [], "unconverged": [],
                "floors": {}, "curve": []})
            cap = int(point["cap"])
            if point.get("status") == "refused":
                row["refused"].append(cap)
                continue
            if point.get("status") == "refused-memory":
                row["refused_memory"].append(cap)
                continue
            if point.get("status") == "unconverged":
                row["unconverged"].append(cap)
                continue
            if point.get("status") != "ok":
                continue
            if "grade" not in point:
                # An oracle-free ladder (a rung past the conventional-CI ceiling): it has a
                # reduction and an internal-convergence verdict, but no tier, and a lane
                # with no tier has no floor to report here.
                row.setdefault("no_oracle", []).append(cap)
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


def summarize_relax(stages: Sequence[str]) -> List[Dict]:
    """Protocol B beside protocol A, per (system, cap) — the amplify-or-absorb answer.

    ⚠ **The two tiers are not interchangeable and the table keeps them apart.** Protocol A
    grades a truncated CI at the exact solve's own orbitals; protocol B grades the
    calculation a user would actually run, where the truncated RDMs move the orbitals too.
    A cap whose tier is worse under B than under A is one where relaxation *amplified* the
    truncation error; the same tier means it was absorbed. A refusal under B and a tier
    under A is the sharpest outcome of all: the truncated ensemble is measurable but not
    optimizable.
    """
    rows: List[Dict] = []
    for stage in stages:
        path = camp.RECORDS / "{}.json".format(stage)
        if not path.is_file():
            continue
        data = json.loads(path.read_text())
        legs = {j["key"]: j for j in data.get("jobs", [])
                if j.get("stage_kind") == "relax"}
        for point in data.get("points", []):
            if point.get("protocol") != "B" or point.get("cap") is None:
                continue
            leg = legs.get(point["key"], {})
            a = point.get("protocol_a") or {}
            rows.append({
                "stage": stage, "key": point["key"], "topology": point["topology"],
                "cap": int(point["cap"]), "status": point["status"],
                "grade_b": point.get("grade", {}).get("overall"),
                "grade_a": a.get("grade") or a.get("status"),
                "converged": point.get("converged"),
                "iterations": point.get("iterations"),
                "e_casscf_error_eh": point.get("e_casscf_error_eh"),
                "max_energy_dev_cm": point.get("grade", {}).get("max_energy_dev_cm"),
                "max_g_rel_dev": point.get("grade", {}).get("max_g_rel_dev"),
                "cpu_s": point.get("cpu_s"),
                "ci_casscf_cpu_s": leg.get("ci_leg", {}).get("cpu_s"),
                "cpu_ratio_to_ci_casscf": point.get("cpu_ratio_to_ci_casscf"),
                "error": point.get("error")})
    rows.sort(key=lambda r: (r["key"], r["topology"], r["cap"]))
    return rows


def print_relax_summary(rows: Sequence[Dict]) -> None:
    if not rows:
        return
    head = ("{:<12s} {:>5s} {:>5s} {:>6s} {:>15s} {:>15s} {:>13s} {:>9s} {:>7s}"
            .format("system", "topo", "D", "iters", "protocol A", "protocol B",
                    "dE_CASSCF/Eh", "CPU s", "xCI"))
    print("\n" + head)
    print("-" * len(head))
    for r in rows:
        b = r["grade_b"] or r["status"]
        print("{:<12s} {:>5s} {:>5d} {:>6s} {:>15s} {:>15s} {:>13s} {:>9s} {:>7s}".format(
            r["key"], r["topology"], r["cap"], str(r["iterations"] or "-"),
            str(r["grade_a"] or "-"), b,
            "-" if r["e_casscf_error_eh"] is None
            else "{:+.3e}".format(r["e_casscf_error_eh"]),
            "-" if r["cpu_s"] is None else "{:.0f}".format(r["cpu_s"]),
            "-" if r["cpu_ratio_to_ci_casscf"] is None
            else "{:.0f}".format(r["cpu_ratio_to_ci_casscf"])))


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
            (",".join(str(c) for c in sorted(r["refused"])) or "-")
            + ("  mem:" + ",".join(str(c) for c in sorted(r["refused_memory"]))
               if r.get("refused_memory") else "")))


# --- driver ---------------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True,
                    choices=sorted(STAGES) + ["summary", "regrade", "extrapolate"])
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

    if args.stage == "extrapolate":
        rows = []
        for stage in sorted(k for k in STAGES if k.startswith("s1")):
            path = camp.RECORDS / "{}.json".format(stage)
            if not path.is_file():
                continue
            data = json.loads(path.read_text())
            lanes = {(p["key"], p["topology"]) for p in data.get("points", [])
                     if p.get("status") == "ok" and p.get("cap") is not None
                     and p.get("protocol") != "B"}
            for key, topology in sorted(lanes):
                caps = sorted(int(p["cap"]) for p in data.get("points", [])
                              if p.get("key") == key and p.get("topology") == topology
                              and p.get("status") == "ok" and p.get("cap") is not None)
                # ⚠ **The whole ladder is the wrong series to extrapolate and is reported
                # only to show that it is.** A straight line in w_disc across weights
                # spanning 0.7 down to 1e-4 is a fit to the unusable end; the extrapolation
                # a production run would actually do uses the last few points, where the
                # error has some chance of being linear in the discarded weight.
                subsets = ([caps] if args.caps is None else [list(args.caps)])
                if args.caps is None:
                    subsets += [caps[-3:], caps[-2:]]
                seen = set()
                for subset in subsets:
                    if len(subset) < 2 or tuple(subset) in seen:
                        continue
                    seen.add(tuple(subset))
                    got = extrapolate(stage, key, topology, subset)
                    if got is not None:
                        rows.append(got)
        camp.RECORDS.mkdir(parents=True, exist_ok=True)
        out_path = camp.RECORDS / "extrapolation.json"
        with open(out_path, "w") as fh:
            json.dump({"schema": SCHEMA, "rows": rows}, fh, indent=1, sort_keys=True,
                      default=_jsonable)
        head = "{:<14s} {:>5s} {:<22s} {:>10s} {:>12s}  {:<14s}  {}".format(
            "system", "topo", "caps", "CPU s", "dE_SA [cm^-1]", "extrapolated", "per point")
        print("\n" + head)
        print("-" * len(head))
        for r in rows:
            print("{:<14s} {:>5s} {:<22s} {:>10.0f} {:>12.3f}  {:<14s}  {}".format(
                r["key"], r["topology"], ",".join(str(c) for c in r["caps"]),
                r["cpu_s_total"], r["e_sa_error_cm_extrapolated"],
                r["grade"]["overall"], ",".join(g[:4] for g in r["per_point_grade"])))
        print("\n-> {}".format(out_path))
        return 0

    if args.stage == "summary":
        stages = sorted(k for k in STAGES if k.startswith("s1"))
        rows = summarize(stages)
        relax = summarize_relax(stages)
        camp.RECORDS.mkdir(parents=True, exist_ok=True)
        path = camp.RECORDS / "summary.json"
        with open(path, "w") as fh:
            json.dump({"schema": SCHEMA, "bands": camp.BANDS.__dict__, "lanes": rows,
                       "protocol_b": relax},
                      fh, indent=1, sort_keys=True, default=_jsonable)
        print_summary(rows)
        print_relax_summary(relax)
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
            elif plan["kind"] == "controls":
                stage_controls(key, topologies, record, heartbeat, deadline=deadline,
                               caps=(caps if caps is not None
                                     else plan["control_caps"][key]),
                               variants=plan["control_variants"],
                               max_sweeps=args.max_sweeps)
            elif plan["kind"] == "relax":
                stage_relax(key, topologies, record, heartbeat, deadline=deadline,
                            caps=(caps if caps is not None
                                  else plan["relax_caps"][key]),
                            wall_budget=plan["relax_budget"][key],
                            max_iter=plan["relax_max_iter"][key],
                            max_sweeps=args.max_sweeps,
                            drivers=plan.get("relax_drivers", ("trust-region",)))
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
