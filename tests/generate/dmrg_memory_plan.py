"""Where a tensor-network run's memory actually goes, phase by phase, against what it declared.

⚠ **A study generator, not a reference generator.** Everything it writes lands in ``temp/``;
nothing here is committed reference data and nothing in the test suite reads it.

The question
------------
The resource budget is meant to refuse an over-large run *before* it allocates, with a
diagnosis naming the knob. For the dense stages it does. For the tensor-network layer it does
not: a network CASCI whose declared plan sits comfortably inside a 4 GB limit has been killed
by the kernel past 12 GB resident, with no message at all. That failure is invisible to the
ledger by construction — the ledger only sees what a ``reserve``/``require`` call declares —
so the only way to find the missing arrays is to measure the **process**, finely enough to
say which phase of a sweep they belong to.

Instrument
----------
A sampler thread reads ``/proc/self/statm`` every :data:`SAMPLE_S` seconds and charges each
sample to the innermost *phase* currently on a stack. Phases are pushed by wrapping the six
entry points a sweep passes through — the TTNO compile and its per-integral refill, the random
state, an environment build, the two-site Davidson solve, the truncating split, and the RDM
contraction — so every sample belongs to exactly one of them, and whatever is left over is
charged to ``other`` rather than to a bucket that hides it.

⚠ **The wrapping lives here and not in the library.** It is measurement scaffolding: a
production run must not pay a sampler thread, and the phases a fix eventually declares are
sizing functions, not instrumentation.

Three numbers per phase, and the third is the point: the **peak RSS** reached inside it
(above the baseline taken once the integrals exist, so the front end is not charged to the
network), the **declared** resident total the ledger held at that moment, and their
difference — the memory the limit is not protecting.

The axes
--------
``--stage roots`` is the ladder the finding asks for: a fixed active space, a fixed topology
and a fixed bond cap, with only the **root count** moving. ⚠ The topology is deliberately
held at the one the largest root count needs (:func:`dmrg_campaign.node_partition` picks the
partition from ``n_roots``, so letting it move would confound the two axes). ``--stage caps``
moves the bond cap at fixed roots, which is the axis the layer's own sizing functions are
written against.

Run discipline
--------------
Bounded by construction (the ten-minute rule): one sweep per point, a hard wall budget checked
between points, one JSON record written as each point completes, and a heartbeat line per
point so progress is read from the run's own file rather than from a command line.

Usage::

    python tests/generate/dmrg_memory_plan.py --stage roots           # ~6 min
    python tests/generate/dmrg_memory_plan.py --stage caps --budget 900

Records: ``temp/dmrg_memory_plan/<stage>.json``; log: ``temp/dmrg_memory_plan/<stage>.log``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dmrg_campaign as camp                                            # noqa: E402
from progress import Heartbeat                                          # noqa: E402

OUT = REPO / "temp/dmrg_memory_plan"

#: Sampling interval [s]. Fast enough to catch a transient that lives for one Davidson
#: iteration, cheap enough that the sampler costs a fraction of a percent of one core: a
#: ``statm`` read is one open/read/close of a pseudo-file.
SAMPLE_S = 0.005

#: Page size for ``/proc/self/statm``, whose fields are in pages.
PAGE_BYTES = float(os.sysconf("SC_PAGE_SIZE"))

#: The system every stage runs on: TiF3 with the 3d + 4d' double shell, CAS(1, 20 spinors).
#: ⚠ Chosen because its *CI* side is trivial (20 determinants, an exact oracle in 0.13 CPU s)
#: while its *network* side is one of the two that were killed by the kernel — so anything
#: measured here is the network layer and not a conventional-CI residency wearing its name.
SYSTEM = "tif3_dd"

#: One sweep per point. The phases repeat per bond, so a peak is reached inside the first
#: sweep; a second sweep measures the same arrays again at the cost of doubling the ladder.
MAX_SWEEPS = 1


# --- the sampler ---------------------------------------------------------------------------

def rss_gb() -> float:
    """Resident set [GB] of this process, from ``/proc/self/statm`` (field 2, in pages)."""
    with open("/proc/self/statm") as fh:
        return int(fh.read().split()[1]) * PAGE_BYTES / 1024.0 ** 3


class PhaseSampler(object):
    """Peak RSS per phase, sampled from a thread and charged to the innermost phase.

    The phase stack is a plain list guarded by the GIL: a push and a pop are single
    bytecodes on a list, and the sampler only ever reads its last element, so no lock is
    needed and the sampler never blocks the calculation.
    """

    def __init__(self, budget) -> None:
        self._budget = budget
        self._stack: List[str] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.baseline = 0.0
        #: phase -> {"peak_gb", "declared_at_peak_gb", "calls", "seconds"}
        self.phases: Dict[str, Dict[str, float]] = {}

    # -- lifecycle --
    def start(self) -> None:
        self.baseline = rss_gb()
        self._thread = threading.Thread(target=self._run, name="rss-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def reset(self) -> None:
        """Forget the per-phase table, keeping the thread and the baseline."""
        self.phases = {}

    # -- the phase stack --
    @contextmanager
    def phase(self, name: str):
        self._stack.append(name)
        rec = self.phases.setdefault(name, {"peak_gb": 0.0, "declared_at_peak_gb": 0.0,
                                            "calls": 0, "seconds": 0.0})
        rec["calls"] += 1
        t0 = time.time()
        try:
            yield
        finally:
            rec["seconds"] += time.time() - t0
            self._stack.pop()

    def _run(self) -> None:
        while not self._stop.wait(SAMPLE_S):
            try:
                rss = rss_gb()
            except OSError:                                  # pragma: no cover - defensive
                continue
            name = self._stack[-1] if self._stack else "other"
            rec = self.phases.setdefault(name, {"peak_gb": 0.0,
                                                "declared_at_peak_gb": 0.0,
                                                "calls": 0, "seconds": 0.0})
            if rss > rec["peak_gb"]:
                rec["peak_gb"] = rss
                rec["declared_at_peak_gb"] = self._budget.resident_gb()

    # -- the report --
    def table(self) -> List[Dict]:
        rows = []
        for name, rec in sorted(self.phases.items(),
                                key=lambda kv: -kv[1]["peak_gb"]):
            rows.append({"phase": name,
                         "peak_rss_gb": round(rec["peak_gb"], 4),
                         "over_baseline_gb": round(rec["peak_gb"] - self.baseline, 4),
                         "declared_gb": round(rec["declared_at_peak_gb"], 4),
                         "unaccounted_gb": round(rec["peak_gb"] - self.baseline
                                                 - rec["declared_at_peak_gb"], 4),
                         "calls": int(rec["calls"]),
                         "seconds": round(rec["seconds"], 2)})
        return rows


# --- instrumentation (measurement scaffolding; see the module docstring) --------------------

def instrument(sampler: PhaseSampler) -> None:
    """Wrap the six entry points a sweep passes through, in place."""
    from kuiva.dmrg import density, sweep, ttno

    def wrap_function(module, name, phase):
        original = getattr(module, name)

        def wrapped(*args, **kwargs):
            with sampler.phase(phase):
                return original(*args, **kwargs)
        wrapped.__name__ = getattr(original, "__name__", name)
        setattr(module, name, wrapped)

    def wrap_method(cls, name, phase):
        original = getattr(cls, name)

        def wrapped(self, *args, **kwargs):
            with sampler.phase(phase):
                return original(self, *args, **kwargs)
        wrapped.__name__ = name
        setattr(cls, name, wrapped)

    wrap_method(ttno.TTNOTemplate, "__init__", "ttno compile")
    wrap_method(ttno.TTNOTemplate, "fill", "ttno fill")
    wrap_function(sweep, "random_state", "state prep")
    wrap_method(sweep.EnvironmentCache, "_build", "environment build")
    wrap_function(sweep, "_solve_local", "two-site solve")
    # Inside the two-site solve, so the sampler charges an ``apply`` sample to ``apply`` and
    # the rest of the solve to its parent: one H_eff application is the object the whole
    # phase is built around, and "the solve is large" and "one matvec is large" call for
    # different fixes.
    wrap_method(sweep._LocalProblem, "apply", "two-site: apply")
    wrap_method(sweep._LocalProblem, "diagonal", "two-site: diagonal")
    wrap_function(sweep, "_commit_split", "truncation split")
    wrap_function(density, "network_rdms", "rdm contraction")
    # ⚠ The solver reaches network_rdms through its own module-level import, so rebinding it
    # in `density` alone would leave the RDM phase unmeasured — an instrument that silently
    # measures nothing looks exactly like a phase that costs nothing.
    from kuiva.dmrg import solver as solver_mod
    solver_mod.network_rdms = density.network_rdms


# --- the run -------------------------------------------------------------------------------

def _setup_logging(stage: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "{}.log".format(stage)
    handler = logging.FileHandler(str(path), mode="a")
    handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger("kuiva")
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    return path


def build_case(sampler: PhaseSampler):
    """Front end, converged orbitals from the campaign checkpoint, active integrals.

    ⚠ The orbitals are **read** from the campaign's CASSCF checkpoint rather than
    re-optimized: this measures the network layer at a fixed, converged chart, and paying an
    orbital optimization to get there would put the ten-minute rule out of reach for no gain
    in what is being measured.
    """
    from kuiva.io.checkpoint import read_checkpoint
    from kuiva.mcscf.orbopt import CASIntegrals

    system = camp.get(SYSTEM)
    reference = camp.build_reference(system)
    path = camp.orbital_checkpoint(system)
    if not path.is_file():
        raise SystemExit("no converged orbitals at {}: run the campaign's orbital stage "
                         "for {} first".format(path, SYSTEM))
    stored = read_checkpoint(str(path), check_fingerprint=False)
    ints = CASIntegrals.build(reference.factors, reference.h_one_electron(),
                              np.ascontiguousarray(stored.coeff), stored.spaces,
                              e_nuc=reference.data.e_nuc)
    return system, ints


def run_point(system, ints, graph, *, n_roots: int, cap: int,
              sampler: PhaseSampler, template=None) -> Dict:
    """One bounded network solve, phase-resolved. Refusals and failures are records.

    ⚠ ``template`` is the compiled TTNO shared across the whole ladder. It is injected into
    the solver's own cache because a recompile costs ~85 s here and depends on neither the
    root count nor the cap — paying it per rung would put the ladder outside the ten-minute
    rule and would measure the same compile four times. The compile's own phase row is
    therefore recorded once, on the rung that built it.
    """
    from kuiva.dmrg import DMRGSolver
    from kuiva.util import resources as res
    from kuiva.util.errors import SolverFailure

    sampler.reset()
    solver = DMRGSolver(system.n_active_elec, max_bond=int(cap), n_roots=int(n_roots),
                        graph=graph, max_sweeps=MAX_SWEEPS, conv_tol=1e-6,
                        davidson_tol=1e-6, on_split="warn")
    if template is not None:
        solver._templates[graph] = template
    t0, c0 = time.time(), time.process_time()
    status = "ok"
    carried = res.BUDGET.resident_gb()
    try:
        # ⚠ A phase scope, not res.clear(): the compiled TTNO's own reservations were made
        # before this point and must stay on the ledger (they are part of what a network
        # solve holds), while everything this rung reserves goes out of scope with it —
        # otherwise the n-th rung is judged against a limit its predecessors filled.
        with res.BUDGET.in_phase("ladder point"):
            solver.solve(ints)
    except SolverFailure as exc:
        # ⚠ Expected and not a failure of this measurement: one sweep does not converge a
        # network, and every array whose size is in question has been allocated by then.
        status = "unconverged (expected at {} sweep)".format(MAX_SWEEPS)
        del exc
    except (ValueError, res.MemoryLimitError) as exc:
        status = "{}: {}".format(type(exc).__name__, exc)[:200]
    point = {"n_roots": int(n_roots), "cap": int(cap), "status": status,
             "wall_s": round(time.time() - t0, 2),
             "cpu_s": round(time.process_time() - c0, 2),
             "baseline_gb": round(sampler.baseline, 4),
             "carried_declared_gb": round(carried, 4),
             "declared_peak_gb": round(res.BUDGET.peak_gb, 4),
             "phases": sampler.table()}
    point["peak_rss_gb"] = round(max([p["peak_rss_gb"] for p in point["phases"]] or [0.0]), 4)
    point["peak_over_baseline_gb"] = round(point["peak_rss_gb"] - sampler.baseline, 4)
    # ⚠ Against the ledger's RESIDENT total at the instant of the peak, not against
    # ``BUDGET.peak_gb``: that one is a process-wide high-water mark folding in every
    # transient ever *checked*, so dividing by it compares this rung with the whole run.
    resident = max([p["declared_gb"] for p in point["phases"]] or [0.0])
    point["declared_at_peak_gb"] = round(resident, 4)
    point["ratio_peak_to_declared"] = (
        None if resident <= 0.0
        else round(point["peak_over_baseline_gb"] / resident, 2))
    return point


def print_point(point: Dict) -> None:
    print("  roots {:3d}  D {:3d}  wall {:6.1f}s  peak {:6.3f} GB over baseline  "
          "declared {:6.3f} GB  ratio {}  [{}]".format(
              point["n_roots"], point["cap"], point["wall_s"],
              point["peak_over_baseline_gb"], point["declared_at_peak_gb"],
              point["ratio_peak_to_declared"], point["status"]), flush=True)
    for row in point["phases"]:
        if row["over_baseline_gb"] <= 0.0005 and row["phase"] != "other":
            continue
        print("      {:<20s} peak {:6.3f}  declared {:6.3f}  unaccounted {:6.3f}  "
              "{:4d} calls  {:7.1f}s".format(
                  row["phase"], row["over_baseline_gb"], row["declared_gb"],
                  row["unaccounted_gb"], row["calls"], row["seconds"]), flush=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", default="roots", choices=("roots", "caps"))
    ap.add_argument("--budget", type=float, default=600.0,
                    help="hard wall budget [s]; the ladder stops between points")
    ap.add_argument("--cap", type=int, default=16, help="bond cap for the roots ladder")
    ap.add_argument("--roots", type=int, default=10, help="root count for the caps ladder")
    args = ap.parse_args(argv)

    _setup_logging(args.stage)
    OUT.mkdir(parents=True, exist_ok=True)
    record_path = OUT / "{}.json".format(args.stage)

    from kuiva.util import resources as res
    res.ensure_configured()

    sampler = PhaseSampler(res.BUDGET)
    instrument(sampler)
    sampler.start()
    heartbeat = Heartbeat("dmrg_memory_plan_{}".format(args.stage),
                          budget_seconds=args.budget)
    deadline = time.time() + args.budget

    system, ints = build_case(sampler)
    n = int(ints.spaces.n_active)
    # ⚠ The topology is fixed at the one the LARGEST root count of the ladder needs, so the
    # root axis moves alone: node_partition reads n_roots, and letting it move would make
    # every rung a different network as well as a different ensemble.
    top_roots = max(ROOT_LADDER) if args.stage == "roots" else args.roots
    graphs, top_meta = camp.topologies(ints, camp.get(SYSTEM), ["path"])
    graph = graphs["path"]
    sampler.baseline = rss_gb()

    meta = {"system": SYSTEM, "n_active": n, "n_active_elec": system.n_active_elec,
            "stage": args.stage, "max_sweeps": MAX_SWEEPS, "sample_s": SAMPLE_S,
            "topology": top_meta, "topology_roots": top_roots,
            "baseline_gb": round(sampler.baseline, 4),
            "memory_limit_gb": res.BUDGET.limit_gb,
            "threads": os.environ.get("KUIVA_NUM_THREADS", "unset")}
    points: List[Dict] = []
    print("system {} CAS({}, {}) on {} nodes {}; baseline {:.3f} GB, limit {:.1f} GB".format(
        SYSTEM, system.n_active_elec, n, top_meta["n_nodes"], top_meta["node_sizes"],
        sampler.baseline, res.BUDGET.limit_gb))

    from kuiva.dmrg.ttno import TTNOTemplate
    with sampler.phase("ttno compile"):
        template = TTNOTemplate(graph)
    compile_row = [r for r in sampler.table() if r["phase"] == "ttno compile"]
    meta["ttno_compile"] = compile_row[0] if compile_row else None
    print("  ttno compile: peak {:.3f} GB over baseline, declared {:.3f} GB, {:.1f} s"
          .format(meta["ttno_compile"]["over_baseline_gb"],
                  meta["ttno_compile"]["declared_gb"], meta["ttno_compile"]["seconds"]))

    ladder = [(r, args.cap) for r in ROOT_LADDER] if args.stage == "roots" \
        else [(args.roots, c) for c in CAP_LADDER]
    for i, (n_roots, cap) in enumerate(ladder):
        if time.time() > deadline:
            print("  wall budget spent; {} point(s) not run".format(len(ladder) - i))
            break
        point = run_point(system, ints, graph, n_roots=n_roots, cap=cap, sampler=sampler,
                          template=template)
        points.append(point)
        print_point(point)
        heartbeat.tick(i, roots=n_roots, cap=cap,
                       peak_gb=point["peak_over_baseline_gb"])
        record_path.write_text(json.dumps({"meta": meta, "points": points}, indent=1))
    sampler.stop()
    print("records: {}".format(record_path))
    return 0


#: Root counts at fixed cap. Ten is the campaign's state average for this system.
ROOT_LADDER = (1, 2, 4, 10)
#: Bond caps at fixed roots.
CAP_LADDER = (4, 8, 16, 32)


if __name__ == "__main__":
    raise SystemExit(main())
