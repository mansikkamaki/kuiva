"""Re-decide the Cholesky threshold on **energy grounds alone**.

What this is for
----------------
``DEFAULT_CHOLESKY_TOL`` was tightened 1e-6 -> 1e-8 for one reason only: a pivoted
decomposition breaks the spherical symmetry of an atomic basis at the size of the threshold,
and that surfaced as free-ion ``2J+1`` multiplet splittings of 0.23 cm^-1 (Ce3+) and
44.85 cm^-1 (Dy3+). With complete symmetry orbits as pivots the symmetry is exact by
construction, so the threshold goes back to meaning what it always claimed — an error bound on
a two-electron integral — and the number must be re-decided on that basis rather than
inherited from the mitigation.

⚠ **The two questions are separate and this script keeps them separate.** "Does the symmetry
survive a loose threshold?" is answered by ``--phase casscf`` on a free ion; "is a loose
threshold accurate enough?" is answered by ``--phase factor`` and ``--phase casci``, which
never look at a degeneracy at all. Conflating them is how the threshold came to be doing two
unrelated jobs in the first place.

Three phases, in increasing cost:

1. ``factor`` — the factorization against the exact integrals, at fixed orbitals: vector
   count, residual, the SCF electronic energy rebuilt from the factors, and the worst
   active-space ``(tu|vw)`` element error. One SCF per system, then every threshold
   re-factorizes the same ERIs, so the comparison is of the factorization and nothing else.
2. ``casci`` — a CASCI over the same guess orbitals at each threshold. The orbitals are
   produced by PySCF's own SCF and never see our factors, so the energy difference between two
   thresholds **is** the integral error as a correlated energy feels it. ⚠ Its absolute value
   is meaningless for a free ion (the aufbau ROHF guess is not spherical); only differences at
   fixed orbitals are read.
3. ``casscf`` — the observable: a state-averaged two-component CASSCF on a free ion, whose
   ``2J+1`` multiplet spreads must be zero by spherical symmetry. This is minutes per record,
   so run **one system and one threshold per invocation** (the ten-minute rule for ad-hoc runs) and merge.

Run::

    python tests/generate/cholesky_threshold.py --phase factor
    python tests/generate/cholesky_threshold.py --phase casci
    python tests/generate/cholesky_threshold.py --phase casscf --systems ce3p --tols 1e-6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import systems as sysdef                                                     # noqa: E402

from kuiva.integrals.transform import ThreeIndexAO, shell_pair_orbits        # noqa: E402
from kuiva.interface import api                                              # noqa: E402
from kuiva.util import resources as res                                      # noqa: E402

HARTREE_TO_CM = 219474.6313632

#: Hard wall budget per invocation. Everything finished is already on disk.
WALL_BUDGET_S = 9.5 * 60

#: Thresholds swept. 1e-12 is the reference the others are measured against, not a candidate.
TOLS = (1e-4, 1e-6, 1e-8, 1e-10)
REFERENCE_TOL = 1e-12

#: ``2J+1`` manifold sizes of the free-ion ground term ladder, for the spread measurement.
#: Spherical symmetry makes every one of these exactly degenerate.
MANIFOLDS = {"ce3p": (6, 8), "yb3p": (8, 6)}


def _system(key: str) -> sysdef.System:
    for s in sysdef.SYSTEMS:
        if s.key == key:
            return s
    raise SystemExit("no system {!r}".format(key))


def _reference(system: sysdef.System, *, memory_gb: float, screening: str):
    """The scalar X2C reference. Independent of the Cholesky threshold by construction: the
    SCF runs on PySCF's own integrals, so every factorization below sees identical orbitals."""
    molecule = api.Molecule(atoms=system.atoms, basis=system.basis, charge=system.charge,
                            spin=system.spin)
    return api.scalar_x2c_reference(molecule, memory_gb=memory_gb, screening=screening)


def _selection(system: sysdef.System, nelec_total: int) -> Dict:
    """How this system's active space is stated (by orbital character where possible).

    Mirrors :func:`tier2_kuiva.active_space_kwargs`, including its trap: the inactive count
    follows from the **electron count**, never from ``2 * (mo_occ > 0).sum()``.
    """
    if system.active_l:
        return dict(character=(0, system.active_l), n_active=2 * system.ncas,
                    n_active_elec=system.nelecas)
    n_inactive = nelec_total - system.nelecas
    if n_inactive % 2:
        raise SystemExit("{}: {} electrons with {} active leaves an odd inactive count"
                         .format(system.key, nelec_total, system.nelecas))
    return dict(active=list(range(n_inactive, n_inactive + 2 * system.ncas)),
                n_active_elec=system.nelecas)


def _active_coefficients(data, system):
    """AO-basis spinor coefficients of the active space, stated as reproducibility requires."""
    from kuiva.orth.canonical import orthogonalize, project_orbitals
    from kuiva.spinor.expand import expand_scalar_mos
    from kuiva.mcscf.casci import active_space, active_space_by_character

    orth = orthogonalize(data.s_ao, "canonical", report=False)
    mo_work = project_orbitals(orth, data.mo_coeff, data.s_ao)
    spinors = expand_scalar_mos(mo_work, data.mo_energy, data.mo_occ, basis="working",
                                report=False)
    c_ao = spinors.transform_scalar_basis(orth.x, basis="ao").c
    sel = _selection(system, data.nelec_total)
    if "character" in sel:
        space = active_space_by_character(c_ao, data.s_ao, data.ao_layout, data.nelec_total,
                                          atom=0, l=system.active_l, n_pairs=system.ncas,
                                          n_active_elec=system.nelecas,
                                          occupation=spinors.occ)
    else:
        space = active_space(sel["active"], c_ao.shape[1], data.nelec_total,
                             n_active_elec=system.nelecas)
    return c_ao[:, space.spaces.active], space


# --- phase 1: the factorization against exact integrals ------------------------------------

def measure_factorization(key: str, tols, *, memory_gb: float) -> List[Dict]:
    """Vector count, residual, SCF energy error and active-space integral error per threshold.

    The SCF energy error is ``Tr[D (J - K/2)]`` rebuilt from the factors against the same
    quantity from the exact ERIs — the leading energy consequence of the factorization, in Eh,
    on the density the calculation actually has.
    """
    from pyscf import scf

    from kuiva.integrals.transform import assemble_4c, transform_3c

    system = _system(key)
    res.BUDGET.clear()
    data = _reference(system, memory_gb=memory_gb, screening="none")
    orbits = shell_pair_orbits(data.ao_layout.ao_shell, data.ao_layout.ao_atom)

    dm = data.mo_coeff * data.mo_occ @ data.mo_coeff.T
    j_ex, k_ex = scf.hf.dot_eri_dm(data.eri, dm, hermi=1)
    e2_exact = 0.5 * float(np.einsum("ij,ji->", dm, j_ex - 0.5 * k_ex))

    c_act, space = _active_coefficients(data, system)
    rows = []
    for tol in list(tols) + [REFERENCE_TOL]:
        for pivots in (True, False):
            res.BUDGET.clear()
            t0 = time.time()
            factors = ThreeIndexAO.from_eri(data.eri, data.nao, float(tol),
                                            orbits=orbits if pivots else None, report=False)
            wall = time.time() - t0
            lsq = factors.unpack(slice(None))
            j = np.tensordot(np.tensordot(lsq, dm, axes=([1, 2], [0, 1])), lsq, axes=(0, 0))
            k = np.matmul(np.matmul(lsq, dm), lsq).sum(axis=0)
            e2 = 0.5 * float(np.einsum("ij,ji->", dm, j - 0.5 * k))
            eri_act = assemble_4c(transform_3c(factors, c_act, c_act))
            rows.append({"key": key, "tol": float(tol), "orbit_pivots": pivots,
                         "nao": int(data.nao), "naux": factors.naux,
                         "residual": float(factors.residual), "wall_s": round(wall, 3),
                         "e2_error_eh": e2 - e2_exact,
                         "_act": eri_act})
    exact_act = {}
    for row in rows:                       # the tightest factorization is the integral reference
        if row["tol"] == REFERENCE_TOL and row["orbit_pivots"]:
            exact_act = row["_act"]
    for row in rows:
        row["act_eri_error_eh"] = float(np.max(np.abs(row.pop("_act") - exact_act)))
    print("  {:6s} nao={} active={}".format(key, data.nao, space.n_active), flush=True)
    for row in rows:
        print("    tol {:.0e} orbits={:d} naux={:5d} resid={:.2e} dE2={:+.3e} "
              "d(tu|vw)={:.3e} {:.2f} s".format(
                  row["tol"], row["orbit_pivots"], row["naux"], row["residual"],
                  row["e2_error_eh"], row["act_eri_error_eh"], row["wall_s"]), flush=True)
    return rows


# --- phase 2: a correlated energy at fixed orbitals -----------------------------------------

def measure_casci(key: str, tols, *, memory_gb: float) -> List[Dict]:
    """CASCI state energies at each threshold, on **identical** orbitals.

    Everything except the factorization is held fixed, so the energy difference between two
    thresholds is the integral error expressed as a correlated energy.
    """
    system = _system(key)
    res.BUDGET.clear()
    data = _reference(system, memory_gb=memory_gb, screening="none")
    n_states = system.soc_states
    rows = []
    for tol in list(tols) + [REFERENCE_TOL]:
        for pivots in (True, False):
            res.BUDGET.clear()
            reference = api.spinor_reference(data, cholesky_tol=float(tol),
                                             orbit_pivots=pivots, memory_gb=memory_gb)
            out = api.casci(reference, n_states=n_states, report=False,
                            **_selection(system, data.nelec_total))
            energies = np.asarray(out.total_energies, dtype=float)
            rows.append({"key": key, "tol": float(tol), "orbit_pivots": pivots,
                         "naux": reference.factors.naux,
                         "e_states": [float(e) for e in energies],
                         "e_avg": float(energies.mean())})
            print("    casci tol {:.0e} orbits={:d} naux={:5d} e_avg={:.10f}".format(
                tol, pivots, reference.factors.naux, energies.mean()), flush=True)
    ref = [r for r in rows if r["tol"] == REFERENCE_TOL and r["orbit_pivots"]][0]
    ref_e = np.asarray(ref["e_states"])
    for row in rows:
        e = np.asarray(row["e_states"])
        row["e_avg_error_eh"] = float(np.mean(e) - ref_e.mean())
        row["e_state_max_error_eh"] = float(np.max(np.abs(e - ref_e)))
        # relative energies are what a spectrum is read from, and errors partly cancel in them
        row["e_rel_max_error_cm"] = float(np.max(np.abs(
            (e - e.min()) - (ref_e - ref_e.min()))) * HARTREE_TO_CM)
    return rows


# --- phase 3: the observable ----------------------------------------------------------------

def measure_casscf(key: str, tol: float, *, pivots: bool, memory_gb: float, max_iter: int,
                   screening: str) -> Dict:
    """One state-averaged CASSCF record: manifold spreads, which symmetry makes exactly zero."""
    system = _system(key)
    res.BUDGET.clear()
    t0 = time.time()
    molecule = api.Molecule(atoms=system.atoms, basis=system.basis, charge=system.charge,
                            spin=system.spin)
    reference = api.spinor_reference(molecule, memory_gb=memory_gb, screening=screening,
                                     cholesky_tol=float(tol), orbit_pivots=pivots)
    outcome = api.casscf(reference, character=(0, system.active_l),
                         n_active=2 * system.ncas, n_active_elec=system.nelecas,
                         n_states=system.soc_states, max_iter=max_iter, conv_grad=1e-4,
                         report=False)
    energies = np.sort(np.asarray(outcome.ci.total_energies, dtype=float)) * HARTREE_TO_CM
    spreads, start = [], 0
    for size in MANIFOLDS[key]:
        block = energies[start:start + size]
        spreads.append(float(block.max() - block.min()))
        start += size
    rec = {"key": key, "tol": float(tol), "orbit_pivots": pivots, "screening": screening,
           "naux": reference.factors.naux,
           "e_casscf_avg": float(outcome.energy),
           "casscf_converged": bool(outcome.converged),
           "casscf_iterations": int(outcome.orbital.n_iterations),
           "manifold_sizes": list(MANIFOLDS[key]),
           "manifold_spread_cm": spreads,
           "worst_spread_cm": max(spreads),
           "splitting_cm": float(energies[MANIFOLDS[key][0]] - energies[0]),
           "seconds": round(time.time() - t0, 1)}
    print("  {} tol={:.0e} orbits={:d}: spreads {} cm^-1, splitting {:.2f}, {:.0f} s"
          .format(key, tol, pivots, ["{:.4f}".format(s) for s in spreads],
                  rec["splitting_cm"], rec["seconds"]), flush=True)
    return rec


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", default="factor", choices=("factor", "casci", "casscf"))
    ap.add_argument("--systems", default="", help="comma-separated system keys")
    ap.add_argument("--tols", default="", help="comma-separated thresholds")
    ap.add_argument("--no-pivots", action="store_true",
                    help="casscf phase: plain column pivoting, i.e. the pre-Stage-3 behaviour")
    ap.add_argument("--screening", default="x2camf")
    ap.add_argument("--max-iter", type=int, default=60)
    ap.add_argument("--memory-gb", type=float, default=8.0)
    ap.add_argument("--out", default="temp/cholesky_threshold.json")
    args = ap.parse_args(argv)

    res.ensure_configured(args.memory_gb)
    tols = [float(t) for t in args.tols.split(",") if t] or list(TOLS)
    defaults = {"factor": "ne,ticl3,ce3p", "casci": "ne,ticl3,ce3p", "casscf": "ce3p"}
    keys = [k for k in (args.systems or defaults[args.phase]).split(",") if k]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    results: Dict = {"phases": {}}
    if os.path.isfile(args.out):
        results = json.loads(open(args.out).read())
        results.setdefault("phases", {})
    rows = results["phases"].setdefault(args.phase, [])

    def flush():
        with open(args.out, "w") as handle:
            json.dump(results, handle, indent=2, sort_keys=True)

    started = time.time()
    print("cholesky threshold study: phase {} on {} (budget {:.0f} s)"
          .format(args.phase, ",".join(keys), WALL_BUDGET_S), flush=True)
    for key in keys:
        if time.time() - started > WALL_BUDGET_S:
            print("  wall budget reached; stopping before {}".format(key), flush=True)
            break
        if args.phase == "factor":
            rows.extend(measure_factorization(key, tols, memory_gb=args.memory_gb))
        elif args.phase == "casci":
            rows.extend(measure_casci(key, tols, memory_gb=args.memory_gb))
        else:
            for tol in tols:
                rows.append(measure_casscf(key, tol, pivots=not args.no_pivots,
                                           memory_gb=args.memory_gb,
                                           max_iter=args.max_iter, screening=args.screening))
                flush()
        flush()                                       # incremental, within the ten-minute ad-hoc budget
    print("wrote {} ({:.0f} s)".format(args.out, time.time() - started), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
