"""The multiplet-splitting protocol for SC-NEVPT2.

The predeclared question
------------------------
**Does the state-specific second-order correction split states that are exactly degenerate in
the reference, and if so, by how much and through which mechanism?** A ``2J+1`` multiplet of a
free ion is degenerate by spherical symmetry; 0.1 cm^-1 is the size at which a
splitting already implies different physics. The literature reports the effect as severe for
CASPT2 (Granovsky 2011's XMCQDPT2 invariance analysis; Shiozaki et al. 2011's XMS-CASPT2) and
mild for NEVPT2 (Guo, Sivalingam, Neese 2021), which is an expectation about other codes and
not a measurement of *this* one.

The order is the user's directive and it is not negotiable: **verify, then diagnose, then
decide.** Nothing in this script chooses a cure.

What is measured, and what each measurement can fail on
--------------------------------------------------------
Every number below is a *spread within one degenerate manifold of the reference spectrum*, in
cm^-1. The manifolds are read off the **CASSCF** spectrum, so the perturbation is never allowed
to define its own idea of which states ought to be degenerate.

``existence``
    The default run (state-averaged Fock, the default). Per-manifold spread of ``E_CASSCF + E2``, beside
    the spread the *reference* already carries. ⚠ **The reference spread is the floor and it is
    not zero** — a converged CI carries 1e-6..1e-5 cm^-1 of Davidson noise on a degenerate
    block — so a corrected spread of that order is a statement about the CI solver, not about
    the perturbation. Reporting the corrected spread alone would credit the perturbation with
    the reference's noise.

``m1``
    Mechanism M1: *the arbitrary basis inside a degenerate CI block*. The CI returns an
    arbitrary unitary mix of a degenerate manifold; per-state RDMs are not invariant under that
    mix, so per-state ``E2`` is not either. Measured **directly**: seeded Haar unitaries are
    applied to each manifold of the converged CI vectors and the correction recomputed. The
    rotated vectors are still exact eigenvectors with the same eigenvalue, so nothing else in
    the calculation moves and the Davidson noise floor is held fixed by construction — the
    spread across rotations *is* M1, with no subtraction and no modelling.

    ⚠ **Only manifolds of dimension >= 3 carry M1 at all.** Any orthonormal basis of a
    *two*-dimensional degenerate space is a Kramers pair (``T`` maps a member into the space and
    orthogonal to it, since ``T^2 = -1``), so a Kramers doublet's two members are related by a
    symmetry however the eigensolver mixed them and their ``E2`` is equal by theorem. A
    protocol that measured M1 on doublets would measure zero and conclude the wrong thing.

    ⚠ **M1 has an orbital twin that is already fixed** (the group-complete
    contraction): the same arbitrariness
    inside a degenerate ``eps`` block of the canonical orbitals, which moved ``Sir`` by 4e-6
    relative until the contraction was made group-complete. The discriminator is recorded here
    because the two look identical in any spread: the orbital twin shows up with **one** state
    and a fixed CI vector, and this measurement holds the CI vector's *space* fixed while
    changing its basis, which the orbital twin cannot see.

``m2``
    Mechanism M2: *a time-odd zeroth-order Hamiltonian*. ``fock="state-specific"`` builds ``H0``
    from a single state's density, which for an odd-electron state is not time-reversal even, so
    ``H0`` itself breaks Kramers symmetry. Excluded by default — this measures what the
    default is worth.

    ⚠ **This is also the guard that can fail**, and it is the reason ``existence`` is
    interpretable at all. A protocol that reports "no splitting" without ever having produced
    one is indistinguishable from a protocol that cannot detect one (the companion-test
    pattern). M2 is the configuration in which the splitting is *expected*, and the record
    stores the ratio it reaches over the default.

    ⚠ **But M2 cannot split a Kramers DOUBLET, and that is a theorem rather than a small
    number.** The partner of a state in a two-dimensional degenerate space is its time reverse
    whatever basis the eigensolver returned; the time-reversed density builds the time-reversed
    Fock, whose ``eps`` spectrum is identical, so every class energy of the two members agrees.
    ``ticl3``, whose ligand field leaves nothing but doublets, measures 4e-07 cm^-1 where the
    free ions measure tens — the same structure that makes M1 vanish on a doublet. Reading the design's
    rationale as "a state-specific Fock splits Kramers pairs" is therefore too strong; what it
    splits is a manifold **larger** than a pair, and there it does so enormously.

``per_class``
    Which classes carry the spread. A spread confined to the primed classes
    implicates the Koopmans/rank-4 chain; a uniform one implicates the norms and the RDMs
    themselves.

``m4``
    Mechanism M4: *genuine quasi-degeneracy between distinct multiplets*. Not an implementation
    artefact — a state-specific perturbation shifts two nearby multiplets independently, without
    letting them interact, which is the mechanism whose cure is a quasi-degenerate treatment.
    Quantified per adjacent pair of manifolds as ``|differential shift| / reference gap``: at
    1 the correction has moved the pair through each other. It needs no extra calculation, only
    the ``existence`` data, and it is why a **ligand-field** system is in the sweep — free-ion
    multiplets are thousands of cm^-1 apart and cannot exhibit it.

Systems
-------
The **p^4 series O / S / Se / Te** of :mod:`nevpt2_frozen_core` — imported rather than
redefined, since it is the same measurement family and a second copy would drift — plus ``bi``
and ``ticl3`` from ``systems.py``. What each contributes:

* **p^4 atoms**: free ions whose ``3P2`` (5), ``3P1`` (3) and ``1D2`` (5) manifolds are
  degenerate by spherical symmetry, at 15 states = the complete active CI space, so the state
  average is complete by construction and the state average's manifold-boundary hazard is structurally
  absent. Manifolds of 3 and 5 make M1 live. This is the cheapest system in the suite that has
  the structure under test — the cheapest-system rule.
* **``bi``** (6p^3): the main-group atom that is multireference *and* strongly spin-orbit
  coupled, with a 4-fold ``4S3/2`` ground manifold — a degeneracy that is a *Kramers* pair of
  Kramers pairs, i.e. the smallest manifold in which M1 can act at all.
* **``ticl3``** (d^1 in a D3h ligand field): five Kramers doublets separated by hundreds of
  cm^-1 rather than thousands. Kramers' theorem makes every manifold a doublet, so M1 is
  **zero by theorem** here and that is the point — it isolates M4, whose crossing ratios need
  small gaps.

⚠ **The 4f pair (``ce3p``/``yb3p``) and the SMM ion (``dy3p``) are reachable by key and are not
in the default sweep**, exactly as in :mod:`nevpt2_frozen_core`: their state-averaged CASSCF is
of order ten CPU-minutes each and the sweep would exceed the ten-minute ad-hoc budget. The record says which
systems it contains and which were skipped, rather than leaving the reader to assume.

Cost and how this script is bounded
---------------------------------------------
Ordered cheapest first, incremental (the output file is rewritten after every system), with a
hard ``--budget`` checked between systems and a :mod:`progress` heartbeat. One CASSCF per
system, shared by every measurement on it, so a difference between measurements is the
perturbation and nothing else.

Usage::

    python tests/generate/nevpt2_multiplet.py --only o --budget 120       # smoke
    python tests/generate/nevpt2_multiplet.py --budget 540                # the sweep
    python tests/generate/nevpt2_multiplet.py --only ce3p --budget 1800   # the 4f end
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
from nevpt2_frozen_core import SERIES as P4_SERIES                          # noqa: E402
from progress import Heartbeat                                              # noqa: E402

REF_OUT = REPO / "tests/reference/nevpt2_multiplet.json"

#: Schema of the emitted file. Bump when the *meaning* of a stored field changes.
SCHEMA = 1

#: Working-memory limit [GB].
MEMORY_GB = 8.0

#: How wide a window counts as one degenerate manifold of the reference [cm^-1]. Far above the
#: 1e-5 cm^-1 the CI carries and far below any ligand-field splitting (``ticl3``'s smallest gap
#: is ~200 cm^-1), so no choice inside two orders of magnitude changes a single grouping here.
GROUP_TOL_CM = 1.0

#: Seeded rotations per M1 measurement. Haar-distributed, and one identity run beside them.
N_ROTATIONS = 4

#: CASSCF bounds. A budget, not an expectation.
CONV_GRAD = 1.0e-4
MAX_ITER = 40

#: Sweep order: cheapest first, so a structural failure surfaces in the first minute rather
#: than the last.
ORDER = ("o", "s", "bi", "ticl3")


def resolve(key: str) -> sysdef.System:
    """A p^4 measurement atom, or any ``systems.py`` key."""
    return P4_SERIES[key] if key in P4_SERIES else sysdef.get(key)


def haar_unitary(n: int, rng: np.random.Generator) -> np.ndarray:
    """A Haar-distributed U(n), by QR of a complex Ginibre matrix.

    ⚠ The diagonal phase fix is not cosmetic: ``numpy.linalg.qr`` fixes no sign convention, so
    without it the sample is not Haar and a "random rotation" would be biased toward a
    particular basis — which is the very thing this measurement varies.
    """
    z = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / np.sqrt(2.0)
    q, r = np.linalg.qr(z)
    d = np.diagonal(r)
    return q * (d / np.abs(d))


def manifolds_of(energies: np.ndarray, tol_cm: float = GROUP_TOL_CM) -> List[Dict]:
    """``[{start, size, barycentre_cm, spread_cm}, ...]`` over the **reference** spectrum.

    ⚠ The blocks come from the reference and are then applied unchanged to the corrected
    spectrum. Re-grouping the corrected energies would let a manifold the perturbation split
    *become* two manifolds, and the measurement would report zero spread for the split it was
    built to find.
    """
    from kuiva.props.multiplet import HARTREE_TO_CM, degenerate_blocks

    rel = (np.asarray(energies, dtype=float) - float(np.min(energies))) * HARTREE_TO_CM
    order = np.argsort(rel)
    if not np.array_equal(order, np.arange(rel.size)):
        raise SystemExit("the reference energies are not ascending; the block indices this "
                         "record stores would not address the states the caller means")
    return [{"start": int(s), "size": int(n),
             "barycentre_cm": float(np.mean(rel[s:s + n])),
             "spread_cm": float(np.ptp(rel[s:s + n]))}
            for s, n in degenerate_blocks(rel, tol_cm=tol_cm)]


def spreads(values_hartree: np.ndarray, blocks: Sequence[Dict]) -> List[float]:
    """Within-manifold spread of a per-state quantity [cm^-1], one entry per manifold."""
    from kuiva.props.multiplet import HARTREE_TO_CM

    v = np.asarray(values_hartree, dtype=float) * HARTREE_TO_CM
    return [float(np.ptp(v[b["start"]:b["start"] + b["size"]])) for b in blocks]


def barycentres(values_hartree: np.ndarray, blocks: Sequence[Dict]) -> List[float]:
    from kuiva.props.multiplet import HARTREE_TO_CM

    v = np.asarray(values_hartree, dtype=float) * HARTREE_TO_CM
    return [float(np.mean(v[b["start"]:b["start"] + b["size"]])) for b in blocks]


def build_reference(system: sysdef.System, *, basis: Optional[str] = None,
                    memory_gb: float = MEMORY_GB, max_iter: int = MAX_ITER):
    """One state-averaged two-component CASSCF, shared by every measurement on this system."""
    from kuiva.interface import api
    from kuiva.util import resources as res

    # ⚠ `resources.BUDGET` is process-global by design, so in a batch driver it
    # accumulates and a later system is refused against a limit its predecessors filled.
    res.clear()
    basis_name = basis or system.basis
    molecule = api.Molecule(atoms=system.atoms, basis=basis_name, charge=system.charge,
                            spin=system.spin)
    # screening="none": what is measured is a degeneracy the perturbation either preserves or
    # does not, and the two-electron picture change costs a four-component atomic solve that
    # cannot create one.
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
    return reference, outcome, basis_name


def run_system(system: sysdef.System, *, basis: Optional[str] = None,
               memory_gb: float = MEMORY_GB, max_iter: int = MAX_ITER,
               n_rotations: int = N_ROTATIONS, with_m2: bool = True,
               heartbeat=None) -> Dict:
    """The whole measurement protocol on one system."""
    from kuiva.pt.classes import available_classes
    from kuiva.pt.nevpt2 import sc_nevpt2

    t0, c0 = time.time(), time.process_time()
    reference, outcome, basis_name = build_reference(system, basis=basis, memory_gb=memory_gb,
                                                     max_iter=max_iter)
    cas, spaces, coeff = outcome.ci, outcome.active.spaces, outcome.coeff
    h_ao = reference.h_one_electron()
    blocks = manifolds_of(cas.energies)

    record: Dict = {
        "key": system.key, "label": system.label, "basis": basis_name,
        "charge": system.charge, "spin": system.spin, "ncas": system.ncas,
        "nelecas": system.nelecas, "n_states": system.soc_states, "screening": "none",
        "nao": int(reference.data.nao), "nspinor": int(reference.nspinor),
        "n_inactive": int(spaces.n_inactive), "n_virtual": int(spaces.n_virtual),
        "active_space": outcome.active.description,
        "casscf_converged": bool(outcome.converged),
        "casscf_iterations": int(outcome.orbital.n_iterations),
        "casscf_grad_norm": float(outcome.orbital.grad_norm),
        "e_casscf_avg": float(outcome.energy),
        "reference_wall": time.time() - t0, "reference_cpu": time.process_time() - c0,
        "group_tol_cm": GROUP_TOL_CM, "manifolds": blocks,
    }

    def correct(vectors, **kw):
        return sc_nevpt2(reference.factors, h_ao, coeff, spaces, vectors, system.nelecas,
                         energies=cas.energies, e_nuc=reference.data.e_nuc, report=False, **kw)

    # --- 1. existence ------------------------------------------------------------------------
    t, c = time.time(), time.process_time()
    base = correct(cas.vectors)
    record["existence"] = {
        "e2": [float(x) for x in base.e2],
        "reference_spread_cm": [b["spread_cm"] for b in blocks],
        "corrected_spread_cm": spreads(base.total_energies, blocks),
        "corrected_barycentre_cm": [
            x - min(barycentres(base.total_energies, blocks))
            for x in barycentres(base.total_energies, blocks)],
        "per_class_spread_cm": {n: spreads(base.class_energies[n], blocks)
                                for n in available_classes()},
        "wall_seconds": time.time() - t, "cpu_seconds": time.process_time() - c,
    }
    if heartbeat is not None:
        heartbeat.tick(0, system=system.key, stage="existence", energy=float(base.e2[0]))

    # --- 2. M1: the arbitrary basis inside a degenerate CI block --------------------------
    # One run rotates EVERY manifold at once, each by its own independent unitary, so N runs
    # cover every manifold rather than N per manifold.
    live = [b for b in blocks if b["size"] >= 3]
    m1: Dict = {"manifolds": [b["start"] for b in live], "n_rotations": int(n_rotations),
                "note": "manifolds of size < 3 are omitted: a two-dimensional degenerate space "
                        "is a Kramers pair whatever basis the eigensolver returned, so its "
                        "members' E2 are equal by theorem and measuring it would report zero "
                        "for the wrong reason"}
    if live and n_rotations:
        t, c = time.time(), time.process_time()
        members = [np.asarray(base.e2)]
        for trial in range(int(n_rotations)):
            rng = np.random.default_rng(20260809 + trial)
            vectors = np.array(cas.vectors, dtype=np.complex128, copy=True)
            for b in live:
                s, n = b["start"], b["size"]
                vectors[s:s + n] = haar_unitary(n, rng) @ vectors[s:s + n]
            members.append(np.asarray(correct(vectors).e2))
            if heartbeat is not None:
                heartbeat.tick(1 + trial, system=system.key, stage="m1")
        stack = np.array(members)                                   # (1 + n_rot, n_states)
        m1["seeds"] = [20260809 + t_ for t_ in range(int(n_rotations))]
        m1["per_manifold"] = []
        for b in live:
            s, n = b["start"], b["size"]
            sub = stack[:, s:s + n]
            m1["per_manifold"].append({
                "start": s, "size": n,
                # The spread among the members of one rotation: what a user would see.
                "member_spread_cm": spreads_of_rows(sub),
                # How far one slot moves when the basis is re-chosen: M1 itself.
                "across_rotation_cm": float(np.max(np.ptp(sub, axis=0))
                                            * _hartree_to_cm()),
                # The invariant the C1 reporting discipline would rest on.
                "barycentre_spread_cm": float(np.ptp(np.mean(sub, axis=1))
                                              * _hartree_to_cm()),
            })
        m1["wall_seconds"] = time.time() - t
        m1["cpu_seconds"] = time.process_time() - c
    record["m1"] = m1

    # --- 3. M2: the time-odd state-specific Fock (the guard that can fail) ------------------
    if with_m2:
        t, c = time.time(), time.process_time()
        ss = correct(cas.vectors, fock="state-specific")
        record["m2"] = {
            "e2": [float(x) for x in ss.e2],
            "corrected_spread_cm": spreads(ss.total_energies, blocks),
            "per_class_spread_cm": {n: spreads(ss.class_energies[n], blocks)
                                    for n in available_classes()},
            "de2_vs_state_averaged_eh": float(ss.e2[0] - base.e2[0]),
            "wall_seconds": time.time() - t, "cpu_seconds": time.process_time() - c,
        }
        if heartbeat is not None:
            heartbeat.tick(1 + int(n_rotations), system=system.key, stage="m2")

    # --- 4. M4: how far the correction moves adjacent multiplets relative to their gap -------
    ref_bary = [b["barycentre_cm"] for b in blocks]
    cor_bary = barycentres(base.total_energies, blocks)
    cor_bary = [x - cor_bary[0] for x in cor_bary]
    pairs = []
    for k in range(len(blocks) - 1):
        gap = ref_bary[k + 1] - ref_bary[k]
        shift = (cor_bary[k + 1] - cor_bary[k]) - gap
        pairs.append({"lower": blocks[k]["start"], "upper": blocks[k + 1]["start"],
                      "reference_gap_cm": float(gap),
                      "corrected_gap_cm": float(cor_bary[k + 1] - cor_bary[k]),
                      "differential_shift_cm": float(shift),
                      "crossing_ratio": float(abs(shift) / gap) if gap else float("inf")})
    record["m4"] = {
        "pairs": pairs,
        "max_crossing_ratio": max((p["crossing_ratio"] for p in pairs), default=0.0),
        "note": "|differential shift| / reference gap between adjacent manifolds. At 1 the "
                "state-specific correction has moved the pair through each other, which is the "
                "regime a quasi-degenerate treatment exists for. It is not an implementation "
                "artefact and no invariance repairs it",
    }

    record["wall_seconds"] = time.time() - t0
    record["cpu_seconds"] = time.process_time() - c0
    return record


def _hartree_to_cm() -> float:
    from kuiva.props.multiplet import HARTREE_TO_CM
    return HARTREE_TO_CM


def spreads_of_rows(block_energies: np.ndarray) -> List[float]:
    """Within-manifold member spread [cm^-1], one entry per rotation (row 0 is unrotated)."""
    return [float(x) * _hartree_to_cm() for x in np.ptp(block_energies, axis=1)]


def report(record: Dict) -> None:
    print("{}  {}  {} states, {} inactive / {} virtual spinors  [cpu {:.1f} s]".format(
        record["label"], record["basis"], record["n_states"], record["n_inactive"],
        record["n_virtual"], record["cpu_seconds"]))
    print("  {:>6s} {:>5s} {:>14s} {:>14s} {:>14s}".format(
        "start", "size", "E_ref [cm^-1]", "ref spread", "corrected"))
    ex = record["existence"]
    for b, ref_s, cor_s in zip(record["manifolds"], ex["reference_spread_cm"],
                               ex["corrected_spread_cm"]):
        print("  {:>6d} {:>5d} {:>14.3f} {:>14.3e} {:>14.3e}".format(
            b["start"], b["size"], b["barycentre_cm"], ref_s, cor_s))
    m1 = record.get("m1", {})
    for entry in m1.get("per_manifold", []):
        print("  M1 block {:>3d} (size {}): across rotations {:.3e}, barycentre {:.3e} cm^-1"
              .format(entry["start"], entry["size"], entry["across_rotation_cm"],
                      entry["barycentre_spread_cm"]))
    if "m2" in record:
        print("  M2 state-specific: worst manifold spread {:.3e} cm^-1 (default {:.3e})".format(
            max(record["m2"]["corrected_spread_cm"]), max(ex["corrected_spread_cm"])))
    print("  M4 max crossing ratio {:.3e}".format(record["m4"]["max_crossing_ratio"]))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--basis", default=None, help="override the system's basis")
    parser.add_argument("--memory-gb", type=float, default=MEMORY_GB)
    parser.add_argument("--max-iter", type=int, default=MAX_ITER)
    parser.add_argument("--rotations", type=int, default=N_ROTATIONS)
    parser.add_argument("--no-m2", action="store_true",
                        help="skip the state-specific Fock measurement (it rebuilds the "
                             "canonical orbitals once per state and is the expensive half)")
    parser.add_argument("--budget", type=float, default=540.0,
                        help="hard wall budget [s], checked between systems (12.0)")
    parser.add_argument("--merge", action="store_true",
                        help="keep records already in the output file")
    args = parser.parse_args(argv)

    keys = list(args.only) if args.only else list(ORDER)
    records: List[Dict] = []
    if args.merge and REF_OUT.exists():
        records = json.loads(REF_OUT.read_text()).get("records", [])
    beat = Heartbeat("nevpt2_multiplet")
    start = time.time()
    skipped: List[str] = []
    for key in keys:
        if time.time() - start > args.budget:
            skipped.append(key)
            continue
        record = run_system(resolve(key), basis=args.basis, memory_gb=args.memory_gb,
                            max_iter=args.max_iter, n_rotations=args.rotations,
                            with_m2=not args.no_m2, heartbeat=beat)
        report(record)
        records = [r for r in records
                   if not (r["key"] == record["key"] and r["basis"] == record["basis"])]
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
