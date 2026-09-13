"""The automatic active-space bridge round on the first Tier-3 system, ``mn3_linear``.

What this answers
-----------------
Whether the automatic selection's spectrum test keeps the bridging-ligand orbitals of three
high-spin Mn(II) d^5 centres on a path -- the superexchange pathways -- and what each round
costs at a size no conventional CI reaches (the core alone is CAS(15, 30), 1.6e8
determinants). The keep/drop verdict has so far been shown load-bearing only on a Ti(III)
dimer whose probe gap is an order of magnitude above the converged one.

⚠ **Past the ten-minute rule by nature, and run only on explicit request.** The front end
alone is ~25 minutes (a two-stage SCF recipe on 352 AOs, measured by the Tier-3 campaign).
Every verdict is a fixed-orbital measurement (the protocol's, since 2026-09-13) and the
accepted space is pre-optimized once at the end, here for ``--final-max-iter`` iterations
only: the handoff orbitals are not what this run measures.

Stated departures from the stage defaults, all written into the record:

* the probe budget, ``--max-determinants`` (default 40 000): the protocol refuses the default
  6000, which cannot hold the 32 768 determinants of the three sites' Hund configurations;

* ``solver="dmrg"``, ``max_spinors=48``: the conventional-CI budget refuses the core outright,
  and the provisional 40-spinor network cap would drop the second bridge unprobed -- which
  would answer nothing.
* ``max_states=16``: the Hund product floor is 6^3 = 216; capped at the default 64 the probe
  would average over 72 roots. Sixteen states hold the Lieb-Mattis S = 5/2 ground level and the
  S = 3/2 level above it, which is the boundary the question is about.

Every probe is written as it finishes, with a heartbeat, and the wall budget is checked before
a probe is started (a probe cannot be interrupted, so the only honest stop is not to start one).

⚠ **Superseded as a measurement (2026-09-13).** A truncated core of coupled centres is no
longer measured by the protocol, so this round now records "kept (not measurable)" verdicts
and a final probe; why the fixed-orbital cheap CI cannot answer the bridge question here is
measured on stored integrals by ``measure_cheap_ci_hund.py``.

Usage::

    source setup.sh
    python tests/generate/measure_autocas_tier3.py --out temp/autocas_tier3/run.jsonl
"""
import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "generate"))

import dmrg_campaign as camp                                                 # noqa: E402
from progress import Heartbeat                                              # noqa: E402

from kuiva.autocas import protocol as proto                                 # noqa: E402
from kuiva.autocas.probe import ProbeBudget                                 # noqa: E402
from kuiva.util.logging import add_file_handler                             # noqa: E402

KEY = "mn3_linear"
#: ⚠ The bridging atoms are NAMED: each Mn pair is bridged by one mu-hydroxide and two syn-syn
#: formates, and detection by covalent contact finds only the hydroxide O (a formate O touches
#: one metal). Measured on the first run: no pair carries 50 % of its population on that one O
#: (the best-mixing pairs, projection 0.06-0.09 onto the Mn shell, hold 0.27-0.33 of it), so
#: the contact-detected class was refused. Atoms 4-13 bridge Mn1-Mn2, atoms 14-23 Mn2-Mn3.
BRIDGE_12 = tuple(range(4, 14))
BRIDGE_23 = tuple(range(14, 24))
#: ⚠ And the two pathways are POOLED, as equivalent centres are. Measured on the second run:
#: with the bridges named separately every mixing pair carries the same population on each
#: (0.416 / 0.416, 0.428 / 0.428, ...) -- the molecule is centrosymmetric and the orbitals are
#: the symmetric and antisymmetric combinations of the two pathways -- so no pair reaches 50 %
#: on either bridge alone and both classes were refused. One bridge between the two
#: sublattices, the outer Mn (1, 3) and the central Mn (2,), over all twenty bridging atoms, is
#: the statement the orbitals can answer.
BRIDGE_ALL = BRIDGE_12 + BRIDGE_23
BRIDGE_O = tuple(a for a in BRIDGE_ALL if a in (4, 7, 8, 11, 12, 14, 17, 18, 21, 22))
#: ⚠ Since the bridge class detects whole bridging LIGANDS (contact components touching both
#: sites, other centres blocking), the pooled statement no longer needs its atoms named: the
#: first variant is the detection, the second the named twenty atoms it must reproduce.
TARGET_VARIANTS = (
    ["shells", ("bridge", ((1, 3), (2,)))],
    ["shells", ("bridge", ((1, 3), (2,)), BRIDGE_ALL)],
)
TARGETS = TARGET_VARIANTS[0]
MAX_SPINORS = 48
MAX_STATES = 16


def probe_spin_squared(reference, probe_result, n_states: int):
    """``<S^2>`` of the probe's lowest ``n_states`` roots, from single-state 1- and 2-RDMs.

    The cheap CI's states live on an explicit determinant list, so the solvers' own
    ``<S^2>`` (which applies ``S`` through a full-CI excitation map or a network) does not
    reach them. With ``S_k = sum_pq s_pq a+_p a_q`` and the list's convention
    ``gamma_pq = <a+_p a_q>``, ``Gamma_pqrs = <a+_p a+_r a_s a_q>``, normal ordering gives

        <S_k^2> = sum_ps (s_k s_k)_ps gamma_ps + sum_pqrs (s_k)_pq (s_k)_rs Gamma_pqrs,

    summed over k. ⚠ The active part only: the inactive trace of ``s_k`` vanishes for a
    Kramers-paired inactive set, and the out-of-space families are neglected (small for a
    probe of a restricted reference; not measured here). Validated on ``ti2cl6``, whose
    probe roots must come out as a singlet and a triplet.
    """
    import numpy as np
    from kuiva.ci.strings import connections, rdm12
    from kuiva.props.dump import spinor_operator
    from kuiva.spinor.expand import spin_operator

    ci = probe_result.result if probe_result.fixed_orbitals else probe_result.result.ci
    active = np.asarray(probe_result.space.spaces.active, dtype=int)
    s_mo = spinor_operator(np.ascontiguousarray(probe_result.coeff),
                           spin_operator(reference.data.s_ao))
    s_act = [np.ascontiguousarray(sk[np.ix_(active, active)]) for sk in s_mo]
    conn = connections(ci.dets)
    n = min(int(n_states), ci.civecs.shape[1])
    values = []
    for i in range(n):
        gamma, gam2 = rdm12(ci.dets, ci.civecs[:, i], np.ones(1), conn)
        total = 0.0
        for s in s_act:
            total += np.sum((s @ s) * gamma) + np.einsum("pq,rs,pqrs->", s, s, gam2,
                                                         optimize=True)
        values.append(float(np.real(total)))
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--wall", type=float, default=8 * 3600.0)
    parser.add_argument("--max-iter", type=int, default=8)
    parser.add_argument("--final-max-iter", type=int, default=2,
                        help="macro-iterations of the ONE pre-optimization at the end: the "
                             "handoff orbitals are not what this measures")
    parser.add_argument("--max-determinants", type=int, default=40000,
                        help="the probe's determinant budget; ⚠ the default 6000 is below the "
                             "32 768 determinants of the three sites' Hund configurations, and "
                             "the first run's spectra were truncation artefacts because of it")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    add_file_handler(str(out.with_suffix(".log")))
    hb = Heartbeat("autocas_tier3", budget_seconds=args.wall, meta=dict(key=KEY))
    t0 = time.time()

    def record(**fields):
        fields["elapsed_s"] = round(time.time() - t0, 1)
        with out.open("a") as fh:
            fh.write(json.dumps(fields, default=str) + "\n")

    system = camp.get(KEY)
    system = replace(system, scf_options=dict(system.scf_options, atomic_reference=True))
    record(event="start", targets=TARGETS, solver="dmrg", max_spinors=MAX_SPINORS,
           max_states=MAX_STATES, probe_max_iter=args.max_iter,
           probe_max_determinants=args.max_determinants)

    c0 = time.process_time()
    reference = camp.build_reference(system)
    record(event="front end", nao=int(reference.data.nao), nspinor=int(reference.nspinor),
           e_scf=float(reference.data.e_scf), converged=bool(reference.data.converged),
           n_cholesky=int(reference.factors.naux), cpu_s=round(time.process_time() - c0, 1))
    hb.tick(0, stage="front end")
    if not reference.data.converged:
        record(event="refused", reason="the scalar SCF did not converge")
        hb.finish(status="scf-unconverged")
        return 1

    # Diagnostics first, and cheap: which pairs mix with the shell and how much of each sits
    # on each bridge, so that a refusal below still leaves the numbers behind.
    import numpy as np
    from kuiva.autocas import candidates as cand
    from kuiva.autocas import centres as ctr

    centres = ctr.detect_centres(reference, report=False).centres
    construction = cand.shell_candidates(reference, centres, report=False)
    values = np.asarray(construction.avas.eigenvalues, dtype=float)
    shell = set(int(p) for p in np.asarray(construction.shell.columns)[0::2] // 2)
    order = [int(p) for p in np.argsort(-values) if int(p) not in shell][:12]
    for name, atoms in (("both pathways", BRIDGE_ALL), ("both pathways, O only", BRIDGE_O)):
        try:
            pops = cand._fragment_pair_populations(reference, construction.coeff,
                                                   [a - 1 for a in atoms])
            record(event="bridge diagnostic", bridge=name, atoms=list(atoms),
                   pairs=[dict(pair=p, projection=round(float(values[p]), 4),
                               population=round(float(pops[p]), 4)) for p in order])
        except Exception as exc:                       # a diagnostic may not end the run
            record(event="bridge diagnostic failed", bridge=name, reason=repr(exc))

    original = proto.measure
    original_probe = proto.probe
    count = [0]

    def final_probe(ref, coeff, space, **kwargs):
        record(event="final probe start", n_active=int(space.n_active))
        kwargs["budget"] = replace(kwargs["budget"], max_iter=args.final_max_iter)
        result = original_probe(ref, coeff, space, **kwargs)
        record(event="final probe", cpu_s=result.cpu_seconds, wall_s=result.wall_seconds,
               n_determinants=int(result.n_determinants),
               spectrum_cm=[float(x) for x in result.spectrum_cm])
        return result

    def logged(ref, coeff, space, **kwargs):
        if hb.expired:
            record(event="stopped", reason="wall budget spent before probe {}".format(
                count[0] + 1))
            hb.finish(status="budget")
            raise SystemExit(2)
        record(event="measure start", index=count[0] + 1, n_active=int(space.n_active),
               n_elec=int(space.n_elec), n_roots=int(kwargs.get("n_roots", 0)))
        result = original(ref, coeff, space, **kwargs)
        count[0] += 1
        record(event="measure", index=count[0], n_active=int(space.n_active),
               n_elec=int(space.n_elec), n_roots=int(result.n_roots),
               n_determinants=int(result.n_determinants), cpu_s=result.cpu_seconds,
               wall_s=result.wall_seconds, converged=bool(result.converged),
               spectrum_cm=[float(x) for x in result.spectrum_cm],
               description=space.description)
        hb.tick(count[0], n_active=int(space.n_active), cpu=result.cpu_seconds)
        # <S^2> of the roots the target manifold is read from: the Lieb-Mattis check (the
        # ground level must be S = 5/2, <S^2> = 8.75) and what each manifold IS.
        try:
            c1 = time.process_time()
            s2 = probe_spin_squared(ref, result, MAX_STATES)
            record(event="measure spin", index=count[0], s_squared=[round(x, 4) for x in s2],
                   cpu_s=round(time.process_time() - c1, 1))
        except Exception as exc:                       # a diagnostic may not end the run
            record(event="measure spin failed", index=count[0], reason=repr(exc))
        return result

    proto.measure = logged
    proto.probe = final_probe
    assembly = None
    # ⚠ A refusal of the target set is a result, and the reference behind it cost 25 minutes:
    # it is recorded and the next narrower statement is tried on the same reference.
    for targets in TARGET_VARIANTS:
        try:
            assembly = proto.assemble(reference, targets=targets, solver="dmrg",
                                      max_spinors=MAX_SPINORS, max_states=MAX_STATES,
                                      probe_budget=ProbeBudget(
                                          max_iter=args.max_iter,
                                          max_determinants=args.max_determinants),
                                      report=True)
            record(event="assembled", targets=targets)
            break
        except ValueError as exc:
            record(event="refused", targets=targets, reason=str(exc))
    if assembly is None:
        hb.finish(status="refused")
        return 1
    for r in assembly.rounds:
        record(event="round", index=r.index, cls=r.cls, candidates=r.n_candidates,
               pruned=r.n_pruned, d_cm=r.distance_cm, tol_cm=r.tolerance_cm,
               verdict=r.verdict, n_spinors=r.n_spinors, cpu_s=r.cpu_seconds, note=r.note)
    record(event="done", n_active=assembly.n_active, n_elec=int(assembly.space.n_elec),
           floor=assembly.floor, product_floor=assembly.product_floor,
           proposal=assembly.proposal.n_states, proposal_text=assembly.proposal.describe(),
           centres=[c.where for c in assembly.centres], statement=assembly.space.description,
           notes=list(assembly.notes), cpu_s=round(time.process_time() - c0, 1))
    hb.finish(status="done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
