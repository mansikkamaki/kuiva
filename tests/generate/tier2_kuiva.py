"""Kuiva's own Tier-2 record: two-component CASSCF SOC spectra and moment matrices.

What this is for
----------------
``tier2_molcas.json`` and ``tier2_dirac.json`` say what OpenMolcas and DIRAC get. This
generator says what **Kuiva** gets, on the same systems and in the same bases, so that
``tests/test_tier2_soc.py`` can compare them — and so that the two standing debts (a Kuiva-side tolerance band, and a *molecular* accuracy statement for X2CAMF) can be
written against measured numbers rather than asserted.

The protocol, and why each part of it is what the external codes did
---------------------------------------------------------------------
* **State-averaged two-component CASSCF over the whole SOC manifold.** The reference records
  average over every root of the spin-free manifold (``System.nroots``), so Kuiva averages
  over all ``System.soc_states`` spinor roots. ⚠ Averaging over fewer would optimize orbitals
  for a different state and move every excitation energy by thousands of cm^-1 — it is not a
  cheaper version of the same calculation.
* **The active space by orbital character** where ``System.active_l`` says so, and the
  frontier spinor window otherwise, which is what ``tier1_pyscf.select_active`` does.
* **Both bases** (matched bases): the project default and the external code's own. A
  discrepancy is only attributable to the Hamiltonian if the basis is held fixed.
* **X2CAMF screening on** (the default). ⚠ This is where the wall time goes: one
  four-component atomic solve per (element, basis), ~35 minutes for a lanthanide, cached on
  disk and paid once ever. The record stores the Hamiltonian provenance so a later reader
  cannot mistake a screened record for an unscreened one.
* **Everything comparable is phase invariant**: degeneracy patterns, relative
  energies, and ``M_ij = Tr_block(mu_i mu_j)`` with its principal g values, from
  :func:`kuiva.props.multiplet.analyse_spectrum`. No matrix element is ever stored as a
  comparable quantity, because the dump fixes no phase convention.

⚠ Cost, and how this script is designed around it
----------------------------------------------------------
This is **not** a ten-minute run and it is not meant to be: the full sweep is hours, dominated
by the atomic four-component solves. It is therefore built the way the bounded-run rule requires of anything
that long:

* **Incremental.** The output file is rewritten after every record, so a run killed at any
  point keeps everything it finished. Re-running with ``--merge`` resumes.
* **Explicitly bounded.** ``--budget`` is a hard wall budget checked between records, and
  ``--max-iter`` bounds the CASSCF; termination never depends on convergence.
* **Self-reporting.** A :mod:`progress` heartbeat is written per record, so ``python -m
  progress check tier2_kuiva`` distinguishes advancing / grinding / blocked / died without
  ever matching on a command line.
* **Ordered cheapest first**, so the systems that can fail structurally fail in the first
  minute rather than the fifth hour.

Usage::

    python tests/generate/tier2_kuiva.py --only ticl3 --screening none      # smoke, ~1 min
    python tests/generate/tier2_kuiva.py --merge --budget 43200             # the real sweep
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
import thermal                                                              # noqa: E402
from progress import Heartbeat                                              # noqa: E402

REF_OUT = REPO / "tests/reference/tier2_kuiva.json"

#: Schema of the emitted file. Bump when the *meaning* of a stored field changes.
SCHEMA = 1

#: Degeneracy grouping tolerance [cm^-1]; matches the external generators so the degeneracy
#: patterns are formed by the same rule on both sides of the comparison.
DEGENERACY_TOL_CM = 1.0

#: Per-system grouping override [cm^-1]. **Empty, and that is the result of a fix.**
#:
#: It used to carry ``{"dy3p": 100.0}``, because the 126-state average split Dy(3+)'s 2J+1
#: multiplets by 44.85 cm^-1 and the record could not be produced at the physical tolerance at
#: all (the C1 defect). The cause was the state count, not the grouping: 126 is a
#: spin-free sextet count and not a spinor manifold boundary, so it cut a 16-fold manifold in
#: half. ``systems.dy3p`` now averages over **134** and the worst spread is 0.005 cm^-1, so the
#: default 1 cm^-1 grouping resolves every manifold. ⚠ Keep this dict empty unless a *physical*
#: reason appears: an entry here hides a splitting rather than measuring one.
DEGENERACY_TOL_BY_KEY: Dict[str, float] = {}

#: CASSCF convergence. ``conv_grad`` is the orbital gradient norm; 1e-4 is the optimizer's own
#: default and is well inside the cm^-1 scale these spectra are read at.
CONV_GRAD = 1.0e-4
#: Macro-iteration budget. A bound, not an expectation: a record that hits it is
#: written with ``casscf_converged: false`` rather than silently trusted.
MAX_ITER = 60

#: Working-memory limit for the generator [GB]. Overridden by ``--memory-gb``.
#:
#: ⚠ **Back to 8, and the two dimers are the reason it was ever 10.** ``ti2cl6``/``ti2cl6_far``
#: used to be refused before the SCF at 8 GB, on a planned peak of 8.523 GB whose largest term
#: was a **square** ``B^P_pq`` of 4.416 GB — an array the pipeline does not build. Every
#: production path transforms a block: the largest one this sweep builds is 0.301 GB. The
#: pre-flight now plans for the block, ``ti2cl6`` plans at 6.528 GB and peaks at 4.259 GB of
#: real resident set, and the generator no longer needs a limit its own systems taught it to
#: raise. ⚠ Raising a limit is the wrong first response to a refusal — read which array the
#: refusal names first, because a plan can be describing a calculation nobody asked for.
MEMORY_GB = 8.0

#: Order the sweep runs in: cheapest first, so a structural failure surfaces in minutes.
ORDER = ("ticl3", "bi", "tlh", "ce3p", "yb3p", "cecl3", "ti2cl6_far", "ti2cl6", "dy3p")


def ordered_systems(keys: Optional[Sequence[str]] = None):
    """Tier-2 systems, cheapest first, optionally filtered."""
    have = {s.key: s for s in sysdef.SYSTEMS if s.tier2}
    chosen = [k for k in ORDER if k in have]
    chosen += [k for k in sorted(have) if k not in chosen]
    if keys:
        unknown = [k for k in keys if k not in have]
        if unknown:
            raise SystemExit("no Tier-2 system(s) {}; have {}".format(unknown, sorted(have)))
        chosen = [k for k in chosen if k in keys]
    return [have[k] for k in chosen]


def active_space_kwargs(system: sysdef.System, reference) -> Dict:
    """How this system's active space is stated to :func:`kuiva.interface.api.casscf`.

    ``active_l`` set -> **by character** (the only statement an independent
    implementation can reproduce). Otherwise the frontier spinor window, which is the spinor
    equivalent of ``tier1_pyscf.select_active``'s ``[ncore, ncore + ncas)``: with
    ``n_inactive = nelec_total - nelecas`` spinors below it, the window is
    ``[n_inactive, n_inactive + 2*ncas)``.

    ⚠ The electron count fixes the inactive count, never ``2 * (mo_occ > 0).sum()``, and
    an odd inactive count is refused downstream because it would split a Kramers pair.
    """
    n_active_spinor = 2 * system.ncas
    if system.active_l:
        # ⚠ **Every centre of the active element, not just the first.** ``ti2cl6`` has two Ti
        # atoms and one active space spanning both, so a scalar ``"Ti"`` is ambiguous and is
        # refused (rightly — which centre it belongs to is the whole point). The physical
        # statement is "the ten lowest Kramers pairs of d character on the two titaniums".
        element = system.atoms[0][0]
        centres = [i for i, (sym, _) in enumerate(system.atoms) if sym == element]
        return dict(character=(centres, system.active_l),
                    n_active=n_active_spinor, n_active_elec=system.nelecas)
    n_inactive = reference.data.nelec_total - system.nelecas
    if n_inactive % 2:
        raise ValueError("{}: {} electrons with {} active leaves an odd inactive count"
                         .format(system.key, reference.data.nelec_total, system.nelecas))
    window = list(range(n_inactive, n_inactive + n_active_spinor))
    return dict(active=window, n_active_elec=system.nelecas)


def run_system(system: sysdef.System, basis_name: str, *, screening: str,
               memory_gb: float, max_iter: int, mode: str = "auto",
               heartbeat=None, tag: str = "", dump_dir=None) -> Dict:
    """One (system, basis) record: front-end, SA-CASSCF, moment matrices, invariants."""
    from kuiva.interface import api
    from kuiva.props.multiplet import HARTREE_TO_CM, degeneracy_pattern

    from kuiva.util import resources as res

    # ⚠ **Release the previous system's reservations before sizing this one.**
    # ``resources.BUDGET`` is process-global by design (the arrays it accounts for are),
    # so in a batch driver like this one it accumulates across records and a later, *larger*
    # system is refused against a limit its predecessors filled. Observed exactly that: the two
    # Ti2Cl6 records were refused with "already committed 2.105 GB", itemizing arrays from Bi,
    # Tl and Ce — a diagnosis that looks like a genuine memory ceiling and is not. Same reason
    # ``tests/conftest.py`` clears it between tests, and the same idiom.
    res.clear()

    t0 = time.time()
    rec: Dict = {"key": system.key, "label": system.label, "basis": basis_name,
                 "charge": system.charge, "spin": system.spin,
                 "ncas": system.ncas, "nelecas": system.nelecas,
                 "active_l": system.active_l,
                 "nroots": {str(k): v for k, v in system.nroots.items()},
                 "n_soc_states": system.soc_states, "screening": screening,
                 "optimizer_mode": mode}

    molecule = api.Molecule(atoms=system.atoms, basis=basis_name, charge=system.charge,
                            spin=system.spin)
    reference = api.spinor_reference(molecule, memory_gb=memory_gb, screening=screening)
    rec["nao"] = int(reference.data.nao)
    rec["nspinor"] = int(reference.nspinor)
    # ⚠ Factorization provenance. Every number below moves with it: at the same threshold,
    # pivoting on complete symmetry orbits rather than on columns changes Yb(3+)'s free-ion
    # multiplet spreads by 20x, and a record that does not say which was used cannot be
    # compared with one produced the other way.
    rec["cholesky_tol"] = reference.factors.tol
    rec["cholesky_naux"] = int(reference.factors.naux)
    rec["cholesky_orbit_complete"] = bool(reference.factors.orbit_complete)
    rec["e_scf"] = float(reference.data.e_scf)
    rec["scf_converged"] = bool(reference.data.converged)
    if reference.data.soc is not None:
        rec["hamiltonian"] = reference.data.soc.provenance()

    selection = active_space_kwargs(system, reference)
    # ⚠ **Tick per macro-iteration, not per record**. A record here is minutes to
    # hours, so a per-record heartbeat cannot distinguish "grinding through second-order
    # steps" from "deadlocked" — which is exactly the pair no liveness check can
    # separate, and the reason the heartbeat exists at all.
    def beat(info: dict):
        if heartbeat is not None:
            heartbeat.tick(info["iteration"], system=tag or system.key,
                           energy=float(info["energy"]),
                           grad=float(info["grad_norm"]), stage="casscf")
        return None

    outcome = api.casscf(reference, n_states=system.soc_states, max_iter=max_iter,
                         conv_grad=CONV_GRAD, mode=mode, report=False, callback=beat,
                         **selection)
    rec["casscf_converged"] = bool(outcome.converged)
    rec["casscf_iterations"] = int(outcome.orbital.n_iterations)
    rec["casscf_grad_norm"] = float(outcome.orbital.grad_norm)
    rec["e_casscf_avg"] = float(outcome.energy)
    rec["active_space"] = outcome.active.description

    energies = np.asarray(outcome.ci.total_energies, dtype=float)
    rec["e_soc_total_lowest"] = float(energies.min())
    rec["soc_rel_cm"] = [round(float(e), 4) for e in
                         (energies - energies.min()) * HARTREE_TO_CM]

    # ⚠ **Is the state average complete?** The measurement lives in the
    # library — ``api.casscf`` runs it after convergence and warns — so this only records what
    # it found. One implementation, not two: a boundary computed one way here and another way
    # there is exactly the duplication that lets a guard and the thing it guards drift apart.
    # ⚠ ``None`` when the boundary Davidson could not converge its extra roots — a warning in
    # the library, not a failure, so the record has to be able to say "not measured" rather
    # than crash here or, worse, imply a clean boundary that was never established.
    boundary = outcome.boundary
    rec["boundary_measured"] = boundary is not None
    if boundary is not None:
        rec["boundary_margin"] = boundary.margin
        rec["boundary_ndet"] = int(boundary.ndet)
        rec["boundary_spans_full_ci"] = bool(boundary.spans_full_ci)
        rec["boundary_gap_cm"] = boundary.gap_cm
        if not boundary.spans_full_ci:
            rec["boundary_next_cm"] = [round(float(x), 4) for x in boundary.next_cm]
    initial = outcome.boundary_initial
    rec["boundary_initial_gap_cm"] = None if initial is None else initial.gap_cm

    matrices = api.property_matrices(reference, outcome)
    tol_cm = DEGENERACY_TOL_BY_KEY.get(system.key, DEGENERACY_TOL_CM)
    multiplets = matrices.analyse(tol_cm=tol_cm)
    rec["degeneracy_tol_cm"] = tol_cm
    rec["degeneracy_pattern"] = list(degeneracy_pattern(multiplets))
    rec["multiplets"] = [{"size": m.size,
                          "energy_cm": round(float(m.energy_cm), 4),
                          "spread_cm": float(m.spread_cm),
                          "g_values": [round(float(g), 6) for g in m.g_values],
                          "m_tensor": [[round(float(x), 8) for x in row]
                                       for row in m.m_tensor]}
                         for m in multiplets]
    if dump_dir is not None:
        # ⚠ The dump file itself, not just the invariants. It is **not** a reference and is never
        # committed — the dump fixes no phase convention, so nothing in it can be compared element by
        # element. Writing it here exercises the format at every size the suite reaches (134
        # states for dy3p) and makes "runs end to end and writes a dump" a fact rather than a
        # claim.
        path = Path(dump_dir) / "{}-{}.prop".format(system.key, basis_name)
        matrices.write(path, title="{} / {} / screening={}".format(
            system.label, basis_name, screening))
        rec["dump_path"] = str(path)
        rec["dump_bytes"] = int(path.stat().st_size)
    rec["moment_hermiticity"] = float(matrices.hermiticity_error())
    rec["inactive_moment_l"] = [float(x) for x in matrices.inactive_l]
    rec["gauge_origin_choice"] = matrices.origin_label
    rec["status"] = "ok"
    rec["seconds"] = round(time.time() - t0, 2)
    return rec


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Kuiva-side Tier-2 SOC records ")
    ap.add_argument("--only", default="", help="comma-separated system keys")
    ap.add_argument("--bases", default="both", choices=("both", "project", "matched"),
                    help="which basis/bases to run (matched bases)")
    ap.add_argument("--screening", default="x2camf",
                    help="two-electron SOC picture change; 'none' for a fast smoke run")
    ap.add_argument("--budget", type=float, default=12 * 3600.0,
                    help="hard wall budget in seconds, checked between records ")
    ap.add_argument("--max-iter", type=int, default=MAX_ITER)
    ap.add_argument("--mode", default="auto",
                    help="orbital-optimizer step engine: auto | second-order "
                         "| quasi-newton")
    ap.add_argument("--memory-gb", type=float, default=MEMORY_GB)
    ap.add_argument("--merge", action="store_true",
                    help="merge into the existing file and SKIP records already present")
    ap.add_argument("--out", default=str(REF_OUT))
    ap.add_argument("--dump-dir", default="temp/tier2_dumps",
                    help="write the 9 property-matrix files here too (never committed: 9 "
                         "fixes no phase convention, so nothing in them is comparable). "
                         "Empty string disables.")
    args = ap.parse_args(argv)

    import kuiva
    out_path = Path(args.out)
    out: Dict = {"schema": SCHEMA, "generator": "tests/generate/tier2_kuiva.py",
                 "code": "kuiva", "kuiva_version": kuiva.__version__,
                 "degeneracy_tol_cm": DEGENERACY_TOL_CM,
                 "optimizer_mode": args.mode, "conv_grad": CONV_GRAD,
                 "max_iter": args.max_iter,
                 "environment": thermal.describe_environment(), "records": {}}
    if args.merge and out_path.is_file():
        out = json.loads(out_path.read_text())
        out.setdefault("records", {})
        out["environment"] = thermal.describe_environment()

    keys = [k for k in args.only.split(",") if k]
    todo: List = []
    for system in ordered_systems(keys):
        bases = {"both": [system.basis, system.basis_matched],
                 "project": [system.basis],
                 "matched": [system.basis_matched]}[args.bases]
        for basis_name in dict.fromkeys(bases):
            todo.append((system, basis_name))

    heartbeat = Heartbeat("tier2_kuiva", budget_seconds=args.budget,
                          meta={"records": len(todo), "screening": args.screening})
    deadline = time.time() + args.budget
    written = 0
    for index, (system, basis_name) in enumerate(todo):
        tag = "{}/{}".format(system.key, basis_name)
        if args.merge and out["records"].get(tag, {}).get("status") == "ok":
            print("[tier2-kuiva] {:34s} already present, skipping".format(tag), flush=True)
            continue
        if time.time() > deadline:
            # The budget is the design. Stopping here keeps every finished record.
            print("[tier2-kuiva] wall budget of {:.0f} s spent; stopping before {}"
                  .format(args.budget, tag), flush=True)
            break
        with thermal.track_resources() as res:
            try:
                rec = run_system(system, basis_name, screening=args.screening,
                                 memory_gb=args.memory_gb, max_iter=args.max_iter,
                                 mode=args.mode, heartbeat=heartbeat, tag=tag,
                                 dump_dir=args.dump_dir or None)
            except Exception as exc:                                    # noqa: BLE001
                rec = {"key": system.key, "basis": basis_name, "status": "error",
                       "error": "{}: {}".format(type(exc).__name__, exc)}
        rec["resources"] = res.as_dict()
        out["records"][tag] = rec
        # ⚠ Rewritten after every record, not at the end: a run killed at hour five must keep
        # the four hours it finished.
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2, sort_keys=True))
        written += 1
        print("[tier2-kuiva] {:34s} {} pattern={} ({})".format(
            tag, rec.get("status"), rec.get("degeneracy_pattern"), res.summary()), flush=True)
        if rec.get("status") == "error":
            print("  [error] {}".format(rec["error"]), flush=True)
        if res.throttled:
            print("  [warn] {}: CPU thermally clamped for {:.0f}% of the run; its wall time "
                  "reflects cooling, not cost "
                  .format(tag, 100 * (res.throttle_fraction or 0)), flush=True)
        heartbeat.tick(index + 1, system=tag, status=str(rec.get("status")), stage="record")

    n_err = sum(1 for r in out["records"].values() if r.get("status") != "ok")
    print("\nwrote {}  ({} records total, {} written this run, {} not ok)".format(
        out_path, len(out["records"]), written, n_err), flush=True)
    heartbeat.finish(records=len(out["records"]), errors=n_err)
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())
