"""WHY the Loewdin charge fails on the metal halides — the mechanism, isolated.

The characterization (`lowdin_charge_study.py`) established the *fact*: Loewdin puts ~1.2
extra electrons on Ti in TiCl3 relative to Mulliken — enough to flip the sign — and the gap
tracks the ligand, not the metal's diffuseness. Mulliken and Loewdin do not usually disagree
by over an electron, so the fact deserves a mechanism. Three experiments, each isolating one
candidate, all on the same scalar ``sfx2c1e`` densities (PySCF-direct: the partitions need
only ``C``, ``occ``, ``S`` and the AO labels, and the charge arithmetic was already verified
against Kuiva's spinor implementation to 1e-13):

**A. Where does the extra Loewdin population sit?** Per-shell decomposition of
``pop_L - pop_M`` on the metal. If the classic diffuse-tail story holds, the excess lives in
the metal's most diffuse s/p shells — functions with exponents ~0.03, whose radial maximum
sits on top of the ligands, i.e. ligand-region density wearing a metal label. If instead it
spreads over the compact valence shells, the story is different.

**B. The ghost test — the decisive one.** Replace the titanium by a *ghost*: the full Ti
basis at the Ti position, no nucleus and no electrons, over a Cl3(3-) density (three closed
shell chlorides at the TiCl3 geometry). Every electron in this system belongs to a chloride;
whatever population a partition assigns to the ghost centre is, by construction, ligand
density wearing a metal label. If Loewdin hands the ghost an electron-scale population while
Mulliken does not, the TiCl3 sign flip is a labelling artifact of the orthogonalization, not
a statement about the density. (A first attempt deleted the metal's diffuse shells instead;
that experiment is unusable in this basis format — a single PySCF basis entry is a general
contraction whose removal takes core functions with it, moving the SCF by hundreds of
hartree — and the ghost construction asks the same question without touching the basis.)

**C. Overlap scan.** The same charges at stretched geometries (r/r0 up to 3). Both partitions
must converge to the same (ionic-limit) charges as inter-atomic overlap dies; the *rate* says
how much of the gap is overlap-driven. The smallest overlap eigenvalue is recorded with every
row, because near-redundancy is the conditioning of ``S`` by another name.

Run:  ``python tests/generate/lowdin_mechanism_study.py [--out temp/lowdin_mechanism.json]``
Everything is a small scalar SCF; the whole script runs in a few minutes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import sqrtm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from kuiva.basis import registry as reg                                  # noqa: E402
from tests.generate import systems                                       # noqa: E402

L_LETTER = "spdfghi"


def pyscf_basis_name(family: str) -> str:
    return reg.get_family(family).provider_name


def run_scf(atoms, charge: int, spin: int, basis) -> Tuple[object, object]:
    from pyscf import gto, scf
    atom_str = [(a[0].capitalize(), tuple(a[1])) for a in atoms]
    mol = gto.M(atom=atom_str, basis=basis, charge=charge, spin=spin, verbose=0)
    mf = (scf.ROHF(mol) if spin else scf.RHF(mol)).sfx2c1e()
    mf.conv_tol = 1e-10
    mf.kernel()
    return mol, mf


def per_ao_populations(mol, mf) -> Dict[str, np.ndarray]:
    c = np.asarray(mf.mo_coeff)
    d = (c * np.asarray(mf.mo_occ)) @ c.T
    s = mol.intor("int1e_ovlp")
    s_half = np.real(sqrtm(s))
    return {"mulliken": np.einsum("ij,ji->i", d, s),
            "lowdin": np.diag(s_half @ d @ s_half),
            "s_min_eig": float(np.linalg.eigvalsh(s).min())}


def atom_charges(mol, pops: np.ndarray) -> List[float]:
    q = []
    ao_atom = np.array([lbl[0] for lbl in mol.ao_labels(fmt=None)])
    for ia in range(mol.natm):
        q.append(float(mol.atom_charge(ia) - pops[ao_atom == ia].sum()))
    return q


def part_a(key: str, family: str) -> Dict[str, object]:
    """Per-l-channel decomposition of pop_L - pop_M on the heavy atom (atom 0).

    ⚠ Aggregated over each angular-momentum channel, not per shell: the per-shell deltas on
    the f-block metal reach ±6.6 electrons with opposite signs *within one channel* — the
    shells of a channel are strongly overlapping, so a single shell's population is not a
    stable quantity and only the channel sum means anything. That within-channel churn is
    itself recorded (``churn`` = sum of |per-shell delta| over the channel) because it is
    direct evidence of near-redundancy among same-centre functions.
    """
    s0 = systems.get(key)
    mol, mf = run_scf(s0.atoms, s0.charge, s0.spin, pyscf_basis_name(family))
    pops = per_ao_populations(mol, mf)
    delta = pops["lowdin"] - pops["mulliken"]
    labels = mol.ao_labels(fmt=None)          # (atom, symbol, nl, m)
    per_shell: Dict[str, float] = {}
    per_l: Dict[str, Dict[str, float]] = {}
    for i, (ia, _sym, nl, _m) in enumerate(labels):
        if ia != 0:
            continue
        l = nl[-1]
        per_shell[nl] = per_shell.get(nl, 0.0) + float(delta[i])
        per_l.setdefault(l, {"delta": 0.0, "mulliken": 0.0, "lowdin": 0.0})
        per_l[l]["delta"] += float(delta[i])
        per_l[l]["mulliken"] += float(pops["mulliken"][i])
        per_l[l]["lowdin"] += float(pops["lowdin"][i])
    for l, vals in per_l.items():
        vals["churn"] = sum(abs(d) for nl, d in per_shell.items() if nl[-1] == l)
    exps = {}
    for ib in range(mol.nbas):
        if mol.bas_atom(ib) == 0:
            l = L_LETTER[int(mol.bas_angular(ib))]
            e = float(np.min(mol.bas_exp(ib)))
            exps[l] = min(exps.get(l, e), e)
    return {"system": key, "basis": family, "s_min_eig": pops["s_min_eig"],
            "channels": [{"l": l, **vals, "min_exp": exps.get(l)}
                         for l, vals in sorted(per_l.items())]}


def part_b(key: str, family: str) -> Dict[str, object]:
    """The ghost test: the heavy atom's basis with no nucleus and no electrons.

    ``ghost-<sym>`` is PySCF's ghost-atom convention: the functions are present and labeled
    with the metal, the charge and electrons are not. The ligand fragment keeps one closed
    shell per halide (Cl3 gets three extra electrons), so the density is unambiguous.
    """
    s0 = systems.get(key)
    name = pyscf_basis_name(family)
    heavy = s0.atoms[0][0].capitalize()
    ghost_atoms = [("ghost-" + heavy, s0.atoms[0][1])] + list(s0.atoms[1:])
    n_lig = len(s0.atoms) - 1
    mol, mf = run_scf(ghost_atoms, -n_lig, 0, name)
    pops = per_ao_populations(mol, mf)
    ao_atom = np.array([lbl[0] for lbl in mol.ao_labels(fmt=None)])
    ghost = {"system": key, "e_scf": float(mf.e_tot), "converged": bool(mf.converged),
             "s_min_eig": pops["s_min_eig"]}
    for scheme in ("mulliken", "lowdin"):
        ghost[scheme + "_on_ghost"] = float(pops[scheme][ao_atom == 0].sum())
    return ghost


def part_d(key: str, bases: List[str]) -> List[Dict[str, object]]:
    """The 'usually they agree' control: the same molecule in ordinary compact bases.

    Mulliken-vs-Loewdin folklore (a few tenths of an electron at most) comes from compact
    valence bases. Running the same TiCl3 in those bases beside the all-electron X2C family
    turns the folklore into a measured baseline, and ties the gap to the overlap conditioning
    recorded in the same row. These bases go straight to PySCF by name — they are controls
    for the partition arithmetic, not calculations anyone quotes.
    """
    s0 = systems.get(key)
    out = []
    for name in bases:
        mol, mf = run_scf(s0.atoms, s0.charge, s0.spin, name)
        pops = per_ao_populations(mol, mf)
        out.append({"basis": name, "nao": int(mol.nao),
                    "converged": bool(mf.converged), "s_min_eig": pops["s_min_eig"],
                    "q_mulliken": atom_charges(mol, pops["mulliken"]),
                    "q_lowdin": atom_charges(mol, pops["lowdin"])})
    return out


def part_c(key: str, family: str, scales=(1.0, 1.2, 1.5, 2.0, 3.0)) -> List[Dict[str, object]]:
    """Both charges at stretched geometries: the overlap-decay fingerprint."""
    s0 = systems.get(key)
    out = []
    for f in scales:
        atoms = [(sym, tuple(f * x for x in xyz)) for sym, xyz in s0.atoms]
        mol, mf = run_scf(atoms, s0.charge, s0.spin, pyscf_basis_name(family))
        pops = per_ao_populations(mol, mf)
        out.append({"scale": float(f), "e_scf": float(mf.e_tot),
                    "converged": bool(mf.converged), "s_min_eig": pops["s_min_eig"],
                    "q_mulliken": atom_charges(mol, pops["mulliken"]),
                    "q_lowdin": atom_charges(mol, pops["lowdin"])})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="temp/lowdin_mechanism.json")
    ap.add_argument("--family", default="x2c-SVPall-2c")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    data: Dict[str, object] = {}
    data["part_a"] = [part_a("ticl3", args.family), part_a("cecl3", args.family)]
    with open(args.out, "w") as fh:
        json.dump(data, fh, indent=1, sort_keys=True)
    for rec in data["part_a"]:
        print("== per-channel delta(L-M), heavy atom of", rec["system"],
              " s_min_eig %.2e" % rec["s_min_eig"])
        for row in rec["channels"]:
            print("  %s   delta %+7.3f   churn %7.3f   (M %7.3f -> L %7.3f)   "
                  "min exp %.4f" % (row["l"], row["delta"], row["churn"],
                                    row["mulliken"], row["lowdin"], row["min_exp"]))

    data["part_b"] = [part_b("ticl3", args.family), part_b("cecl3", args.family)]
    with open(args.out, "w") as fh:
        json.dump(data, fh, indent=1, sort_keys=True)
    print("== ghost test (ligands only + ghost metal basis)")
    for r in data["part_b"]:
        print("  %-8s converged=%s  electrons on the GHOST:  Mulliken %+7.3f   "
              "Loewdin %+7.3f" % (r["system"], r["converged"],
                                  r["mulliken_on_ghost"], r["lowdin_on_ghost"]))

    data["part_c"] = part_c("ticl3", args.family)
    with open(args.out, "w") as fh:
        json.dump(data, fh, indent=1, sort_keys=True)
    print("== distance scan, TiCl3")
    for r in data["part_c"]:
        print("  r/r0 %.1f  q_M(Ti) %+6.3f  q_L(Ti) %+6.3f  gap %6.3f  s_min %.1e" % (
            r["scale"], r["q_mulliken"][0], r["q_lowdin"][0],
            r["q_mulliken"][0] - r["q_lowdin"][0], r["s_min_eig"]))

    data["part_d"] = part_d("ticl3", ["sto-3g", "def2-svp", pyscf_basis_name(args.family)])
    with open(args.out, "w") as fh:
        json.dump(data, fh, indent=1, sort_keys=True)
    print("== basis control, TiCl3 at equilibrium")
    for r in data["part_d"]:
        print("  %-14s nao %3d  q_M(Ti) %+6.3f  q_L(Ti) %+6.3f  gap %6.3f  s_min %.1e" % (
            r["basis"], r["nao"], r["q_mulliken"][0], r["q_lowdin"][0],
            r["q_mulliken"][0] - r["q_lowdin"][0], r["s_min_eig"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
