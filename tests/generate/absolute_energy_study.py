"""How accurate can an ABSOLUTE total energy be made?

**What this decides.** The Cholesky default was decided on *relative* energies, and the
project's standing statement is that no usable threshold reaches 1e-8 Eh on an absolute
total. That statement bounds nothing: a user cannot tell from it whether an absolute total
is usable at all, or whether tightening the threshold would buy one. This measures the two
error sources that stand between a Kuiva total and the exact answer *in the same basis* —
the factorization, and then the basis itself — so the two can be compared on one scale.

⚠ **The exact four-index reference is the conventional ERI array, not a tighter
factorization.** Phase ``factor`` rebuilds the SCF two-electron energy from the factors and
compares it against ``dot_eri_dm`` on the stored array, and it transforms the active block
both ways (through the factors, and by a direct AO->MO contraction of the same array). Both
are absolute statements against integrals that were never factorized, which is what makes
the threshold series meaningful rather than self-referential. The correlated phases then use
the tightest factorization as their reference, and phase ``factor`` is what licenses that:
it reports how far the tightest factorization itself sits from exact.

Three phases, in increasing cost:

1. ``factor`` — per threshold: vector count, residual, the absolute error of the SCF
   two-electron energy against the exact array, and the worst active ``(tu|vw)`` error
   against the direct transform of the same array.
2. ``casci`` — the absolute CASCI total energy per threshold at **fixed** orbitals, so the
   difference between two thresholds is the integral error as a correlated total feels it.
   Relative (level) energies are carried beside it, because errors cancel there and the
   contrast is the whole point.
3. ``basis`` — the same absolute total in a basis series at a fixed threshold. This is the
   term the threshold is competing against, and it is the reason the answer is what it is.

Run::

    python tests/generate/absolute_energy_study.py --phase factor
    python tests/generate/absolute_energy_study.py --phase casci
    python tests/generate/absolute_energy_study.py --phase basis

One phase per invocation stays inside the ten-minute rule; every phase writes its records
incrementally and merges into the same JSON.
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

from kuiva.integrals.transform import (ThreeIndexAO, shell_pair_orbits,      # noqa: E402
                                       assemble_4c, transform_3c)
from kuiva.interface import api                                              # noqa: E402
from kuiva.util import resources as res                                      # noqa: E402
from progress import Heartbeat                                               # noqa: E402

HARTREE_TO_CM = 219474.6313632

#: Hard wall budget per invocation (the ten-minute rule).
WALL_BUDGET_S = 9.5 * 60

#: Thresholds swept. 1e-14 is the reference for the correlated phases; phase ``factor``
#: measures it too, against integrals that were never factorized.
TOLS = (1e-4, 1e-6, 1e-8, 1e-10, 1e-12)
REFERENCE_TOL = 1e-14

#: Systems: a light closed shell, a molecular light-heavy bond, and a 3d closed shell. All
#: three have an exact four-index reference at this size, which is the membership rule.
KEYS = ("ne", "hi", "zn2p")

#: Basis series for phase ``basis``, smallest first. Karlsruhe segmented, the project default
#: family, so the series is the one a user would actually walk up.
BASES = ("x2c-SVPall-2c", "x2c-TZVPall-2c", "x2c-QZVPall-2c")


def _system(key: str) -> sysdef.System:
    for s in sysdef.SYSTEMS:
        if s.key == key:
            return s
    raise SystemExit("no system {!r}".format(key))


def _reference(system: sysdef.System, *, memory_gb: float, basis: Optional[str] = None):
    """The scalar X2C reference. Independent of the threshold by construction: the SCF runs
    on PySCF's own integrals, so every factorization below sees identical orbitals."""
    molecule = api.Molecule(atoms=system.atoms, basis=basis or system.basis,
                            charge=system.charge, spin=system.spin)
    return api.scalar_x2c_reference(molecule, memory_gb=memory_gb, screening="none",
                                    fitting="cholesky")


def _selection(system: sysdef.System, nelec_total: int) -> Dict:
    """How this system's active space is stated. Mirrors ``cholesky_threshold._selection``,
    including its trap: the inactive count follows from the ELECTRON count."""
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
    """AO-basis spinor coefficients of the active space."""
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


def _spinor_reference_at(data, tol: float, *, memory_gb: float) -> api.SpinorReference:
    """A :class:`SpinorReference` on already-ingested data at the stated threshold.

    ⚠ Built here rather than through ``api.spinor_reference`` for one reason: the stored AO
    integral array is **released** at the first factorization of a container, so factorizing
    the same ingested data at a second threshold refuses. ``release_eri=False`` keeps it,
    which is what makes "identical orbitals, one axis varied" possible at all.
    """
    from kuiva.orth.canonical import orthogonalize, project_orbitals
    from kuiva.spinor.expand import expand_scalar_mos

    res.ensure_configured(memory_gb)
    orth = orthogonalize(data.s_ao, "canonical", report=False)
    mo_work = project_orbitals(orth, data.mo_coeff, data.s_ao)
    spinors = expand_scalar_mos(mo_work, data.mo_energy, data.mo_occ, basis="working",
                                report=False)
    factors = ThreeIndexAO.from_scalar_data(data, float(tol), report=False,
                                            release_eri=False)
    return api.SpinorReference(data=data, orth=orth, spinors=spinors, factors=factors)


def _exact_active_eri(eri: np.ndarray, nao: int, c_act: np.ndarray) -> np.ndarray:
    """``(tu|vw)`` over the active spinors, by direct AO->MO contraction of the EXACT array.

    ⚠ This never touches a factorization, which is the whole point: it is the reference the
    factored active integrals are measured against. The spinors are spin-blocked
    ``(2*nao, n)`` (the fixed convention), and the two-electron operator is spin diagonal, so
    the AO contraction is over the sum of the two spin blocks of each transition density.

    ⚠ ``data.eri`` is PySCF's PACKED array (8-fold symmetry), not ``(nao,)*4``; it is restored
    here rather than indexed, because a packed array indexed as though it were square is the
    kind of mistake that yields a plausible number and no error.
    """
    from pyscf import ao2mo
    eri = np.asarray(ao2mo.restore(1, np.asarray(eri), nao)).reshape(nao, nao, nao, nao)
    c_a, c_b = c_act[:nao, :], c_act[nao:, :]
    # (pq| in the AO pair index, summed over the spin blocks: D^pq_{mu nu}
    d = np.einsum("mp,nq->pqmn", c_a.conj(), c_a) + np.einsum("mp,nq->pqmn", c_b.conj(), c_b)
    n = c_act.shape[1]
    d = d.reshape(n * n, nao * nao)
    g = eri.reshape(nao * nao, nao * nao)
    return (d @ g @ d.T).reshape(n, n, n, n)


# --- phase 1: the factorization against the exact four-index array -------------------------

def measure_factorization(key: str, *, memory_gb: float) -> List[Dict]:
    """Absolute two-electron energy error and absolute active integral error per threshold."""
    from pyscf import scf

    system = _system(key)
    res.clear()
    data = _reference(system, memory_gb=memory_gb)
    if data.eri is None:
        raise SystemExit("{}: no stored ERI array; phase 'factor' needs fitting='cholesky'"
                         .format(key))
    orbits = shell_pair_orbits(data.ao_layout.ao_shell, data.ao_layout.ao_atom)

    dm = data.mo_coeff * data.mo_occ @ data.mo_coeff.T
    j_ex, k_ex = scf.hf.dot_eri_dm(data.eri, dm, hermi=1)
    e2_exact = 0.5 * float(np.einsum("ij,ji->", dm, j_ex - 0.5 * k_ex))

    c_act, space = _active_coefficients(data, system)
    exact_act = _exact_active_eri(data.eri, int(data.nao), c_act)
    scale = float(np.max(np.abs(exact_act)))

    rows = []
    for tol in list(TOLS) + [REFERENCE_TOL]:
        res.clear()
        t0, c0 = time.time(), time.process_time()
        factors = ThreeIndexAO.from_eri(data.eri, data.nao, float(tol), orbits=orbits,
                                        report=False)
        wall, cpu = time.time() - t0, time.process_time() - c0
        lsq = factors.unpack(slice(None))
        j = np.tensordot(np.tensordot(lsq, dm, axes=([1, 2], [0, 1])), lsq, axes=(0, 0))
        k = np.matmul(np.matmul(lsq, dm), lsq).sum(axis=0)
        e2 = 0.5 * float(np.einsum("ij,ji->", dm, j - 0.5 * k))
        act = assemble_4c(transform_3c(factors, c_act, c_act))
        rows.append({
            "key": key, "tol": float(tol), "nao": int(data.nao), "naux": int(factors.naux),
            "n_active": int(space.n_active),
            "residual": float(factors.residual),
            "e2_exact_eh": e2_exact,
            "e2_error_eh": float(e2 - e2_exact),
            "act_eri_max_error_eh": float(np.max(np.abs(act - exact_act))),
            "act_eri_scale_eh": scale,
            "wall_s": round(wall, 3), "cpu_s": round(cpu, 3),
        })
        r = rows[-1]
        print("    tol {:.0e} naux={:5d} resid={:.2e} |dE2|={:.3e} Eh  "
              "max|d(tu|vw)|={:.3e} Eh  {:.2f} s".format(
                  r["tol"], r["naux"], r["residual"], abs(r["e2_error_eh"]),
                  r["act_eri_max_error_eh"], r["wall_s"]), flush=True)
    return rows


# --- phase 2: the absolute correlated total at fixed orbitals -------------------------------

def measure_casci(key: str, *, memory_gb: float) -> List[Dict]:
    """Absolute CASCI total energy per threshold, on identical orbitals."""
    system = _system(key)
    res.clear()
    data = _reference(system, memory_gb=memory_gb)
    n_states = system.soc_states
    rows = []
    for tol in list(TOLS) + [REFERENCE_TOL]:
        reference = _spinor_reference_at(data, float(tol), memory_gb=memory_gb)
        out = api.casci(reference, n_states=n_states, report=False, classify=False,
                        **_selection(system, data.nelec_total))
        energies = np.asarray(out.total_energies, dtype=float)
        rows.append({"key": key, "tol": float(tol), "nao": int(data.nao),
                     "naux": int(reference.factors.naux),
                     "e_states": [float(e) for e in energies]})
        print("    casci tol {:.0e} naux={:5d} E0={:.12f}".format(
            tol, reference.factors.naux, energies[0]), flush=True)
    ref = [r for r in rows if r["tol"] == REFERENCE_TOL][0]
    ref_e = np.asarray(ref["e_states"])
    for row in rows:
        e = np.asarray(row["e_states"])
        row["e_ground_error_eh"] = float(e[0] - ref_e[0])
        row["e_state_max_error_eh"] = float(np.max(np.abs(e - ref_e)))
        row["e_rel_max_error_cm"] = float(np.max(np.abs(
            (e - e[0]) - (ref_e - ref_e[0]))) * HARTREE_TO_CM)
    return rows


# --- phase 3: the basis series, at a fixed threshold ----------------------------------------

def measure_basis(key: str, *, memory_gb: float, tol: float) -> List[Dict]:
    """The same absolute total in a basis series, so the threshold error has a scale to be
    read against. ⚠ The active space is restated by CHARACTER or by electron count in each
    basis, never as a fixed index window, so the series is the same physical calculation."""
    system = _system(key)
    rows = []
    for basis in BASES:
        res.clear()
        t0, c0 = time.time(), time.process_time()
        data = _reference(system, memory_gb=memory_gb, basis=basis)
        reference = _spinor_reference_at(data, float(tol), memory_gb=memory_gb)
        out = api.casci(reference, n_states=system.soc_states, report=False, classify=False,
                        **_selection(system, data.nelec_total))
        energies = np.asarray(out.total_energies, dtype=float)
        rows.append({"key": key, "basis": basis, "tol": float(tol), "nao": int(data.nao),
                     "naux": int(reference.factors.naux),
                     "e_scf": float(data.e_scf),
                     "e_states": [float(e) for e in energies],
                     "wall_s": round(time.time() - t0, 2),
                     "cpu_s": round(time.process_time() - c0, 2)})
        r = rows[-1]
        print("    {:18s} nao={:4d} E_scf={:.10f} E0={:.10f} ({:.1f} s)".format(
            basis, r["nao"], r["e_scf"], r["e_states"][0], r["wall_s"]), flush=True)
    for i in range(1, len(rows)):
        rows[i]["d_e_ground_from_previous_eh"] = (rows[i]["e_states"][0]
                                                  - rows[i - 1]["e_states"][0])
        rows[i]["d_e_scf_from_previous_eh"] = rows[i]["e_scf"] - rows[i - 1]["e_scf"]
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", required=True, choices=("factor", "casci", "basis"))
    ap.add_argument("--out", default="temp/absolute_energy.json")
    ap.add_argument("--systems", default="", help="comma-separated keys (default: all)")
    ap.add_argument("--memory-gb", type=float, default=6.0)
    ap.add_argument("--tol", type=float, default=1e-8, help="phase 'basis': fixed threshold")
    args = ap.parse_args()
    keys = [k for k in (args.systems.split(",") if args.systems else KEYS) if k]

    data: Dict[str, object] = {"reference_tol": REFERENCE_TOL, "tols": list(TOLS),
                               "bases": list(BASES), "records": {}}
    if os.path.exists(args.out):
        with open(args.out) as fh:
            data = json.load(fh)
    data.setdefault("records", {})
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    hb = Heartbeat("absolute_energy_" + args.phase, budget_seconds=WALL_BUDGET_S,
                   meta={"phase": args.phase, "systems": keys})
    for i, key in enumerate(keys):
        if hb.expired:
            hb.tick(i, stage="wall budget exhausted before {}".format(key))
            print("wall budget exhausted before {}".format(key))
            break
        hb.tick(i, stage="{}: {}".format(args.phase, key))
        print("  {} [{}]".format(key, args.phase), flush=True)
        if args.phase == "factor":
            rows = measure_factorization(key, memory_gb=args.memory_gb)
        elif args.phase == "casci":
            rows = measure_casci(key, memory_gb=args.memory_gb)
        else:
            rows = measure_basis(key, memory_gb=args.memory_gb, tol=args.tol)
        data["records"].setdefault(args.phase, {})[key] = rows
        with open(args.out + ".tmp", "w") as fh:
            json.dump(data, fh, indent=1, sort_keys=True)
        os.replace(args.out + ".tmp", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
