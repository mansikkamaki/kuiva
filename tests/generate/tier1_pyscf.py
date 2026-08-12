"""Generate the Tier-1 reference data from PySCF (Tier 1).

Tier 1 is *scalar* validation: PySCF itself supplies the references, so the suite runs on a
laptop with no external quantum-chemistry program. What is locked down here:

* the **scalar X2C SCF** total energy (the ingestion boundary — Kuiva's bridge must
  reproduce this exactly, since it is the same call);
* **CASCI / state-averaged CASSCF** energies and the relative energies of the state manifold;
* **SC-NEVPT2** correlation energies (PySCF's ``mrpt.NEVPT2`` is the strongly contracted
  variant of Angeli et al., i.e. the same method as kuiva.pt).

These are what Kuiva's own CI/CASSCF/NEVPT2 must reproduce **to near machine precision** when
SOC is switched off: a Kramers-paired spinor basis built from real scalar MOs with no
spin-orbit term in the Hamiltonian describes exactly the same problem as a scalar CASSCF, in
a doubled basis. Tier 1 is therefore a much tighter test than the cross-code Tier 2, and it
is the first thing to run when a CI or orbital-optimiser change is suspected.

Reproducibility requirements this script is careful about
---------------------------------------------------------
A stored energy is only a valid reference if an independent implementation can define the
*same* calculation. Two things would otherwise be ambiguous:

1. **Active-orbital selection.** Selecting "the orbitals around the Fermi level" is not
   reproducible for an open-shell heavy atom: the 4f and 5d manifolds interleave, and a
   plain frontier window silently lands on 5d for Ce(3+), moving the state-averaged energy
   by 0.2 Eh and destroying the 7-fold degeneracy that makes the reference meaningful.
   Orbitals are therefore chosen by :func:`select_active`, whose rule is stated in terms of
   physics (*the lowest orbitals of a given angular-momentum character*) rather than of
   PySCF's orbital ordering or AO labelling — see that function for why both halves of the
   rule are needed, and ``systems.System.active_l`` for why angular momentum rather than an
   AO shell label.
2. **Degenerate-orbital mixing.** Within a degenerate shell the individual MOs are arbitrary,
   but CASCI/CASSCF energies are invariant to rotations *inside* the active space — so every
   active space here is a **complete shell**, making the stored energies well defined.

Run:  python tests/generate/tier1_pyscf.py [--fast-only]
(with ``external/env.sh`` sourced). Writes ``tests/reference/tier1_pyscf.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kuiva.basis import registry as reg                      # noqa: E402
from kuiva.props.multiplet import HARTREE_TO_CM              # noqa: E402
import systems as sysdef                                     # noqa: E402
import thermal                                               # noqa: E402

REF_OUT = REPO / "tests/reference/tier1_pyscf.json"

SCF_CONV_TOL = 1e-10
MCSCF_CONV_TOL = 1e-9


# --- helpers ----------------------------------------------------------------------------
def build_mol(system: sysdef.System, basis_name: str, verbose: int = 0):
    """PySCF ``Mole`` for a system in a named basis, resolved through the registry."""
    from pyscf import gto
    syms = list(system.elements)
    if reg.has_family(basis_name) and reg.get_family(basis_name).provider is reg.Provider.BSE:
        basis = reg.resolve_for_pyscf(basis_name, syms)
    else:
        basis = {s: basis_name for s in syms}
    return gto.M(atom=[(s, tuple(xyz)) for s, xyz in system.atoms], basis=basis,
                 charge=system.charge, spin=system.spin, verbose=verbose)


#: Half-width of the candidate window used by :func:`select_active`, in units of ``ncas``.
#: Wide enough for the lanthanide case, where ROHF leaves the singly-occupied 4f at the HOMO
#: but pushes the six empty 4f orbitals above the 5d/6s block (Ce(3+): indices 27, 34-39 with
#: ncore = 27), and narrow enough to exclude diffuse high-l virtuals.
ACTIVE_WINDOW_BELOW, ACTIVE_WINDOW_ABOVE = 1, 4


def lowdin_l_populations(mol, mo: np.ndarray, l_letter: str) -> np.ndarray:
    """Loewdin population of each MO on all AOs of angular momentum ``l_letter``.

    ``pop_k = sum_{mu in l} |(S^{1/2} C)_{mu k}|^2``. Loewdin rather than Mulliken because
    the symmetrically orthogonalised AOs do not overlap: a 5p core MO then carries no
    spurious population on the valence p functions, which is what makes the ranking below
    discriminate core from valence at all.
    """
    import scipy.linalg as sla
    l_idx = "spdfghi".index(l_letter.lower())
    idx = [i for i, lbl in enumerate(mol.ao_labels())
           if lbl.split()[2][1] == "spdfghi"[l_idx]]
    if not idx:
        raise ValueError(f"no AOs of angular momentum {l_letter!r} in this basis")
    s_half = sla.sqrtm(mol.intor("int1e_ovlp")).real
    c = s_half @ mo
    return np.einsum("mk,mk->k", c[idx, :], c[idx, :])


#: Minimum Loewdin population on ``active_l`` for an orbital to count as "of that character".
L_CHARACTER_THRESHOLD = 0.5


def select_active(mol, mo: np.ndarray, mo_energy: np.ndarray,
                  system: sysdef.System) -> List[int]:
    """Indices of the active MOs — deterministic, and reproducible by any implementation.

    With ``active_l == ""`` the frontier window ``[ncore, ncore+ncas)`` is taken, which is
    unambiguous whenever the active shell *is* the frontier shell (Ne, Zn(2+) 3d^10,
    Bi 6p^3, and the sigma/sigma* CAS of HI and TlH).

    Otherwise the rule is **"the lowest-lying orbitals of angular-momentum-``active_l``
    character"**, within a window around the frontier. Two filters, because either alone
    fails on a lanthanide:

    * population alone picks the *diffuse* f polarisation functions, whose f character is
      just as close to 1 as the real 4f shell's;
    * a frontier window alone lands on 5d/6s, because ROHF places the six *empty* 4f
      orbitals above them (Ce(3+): the 4f shell is MOs 27 and 34-39, not 27-33).

    Ranking the f-character orbitals by orbital energy separates compact 4f from diffuse f
    cleanly, and it is a statement about the physics rather than about PySCF's ordering.
    """
    ncore = (mol.nelectron - system.nelecas) // 2
    if not system.active_l:
        return list(range(ncore, ncore + system.ncas))
    nmo = mo.shape[1]
    lo = max(0, ncore - ACTIVE_WINDOW_BELOW * system.ncas)
    hi = min(nmo, ncore + ACTIVE_WINDOW_ABOVE * system.ncas)
    pop = lowdin_l_populations(mol, mo, system.active_l)
    cand = np.arange(lo, hi)
    has_char = cand[pop[cand] > L_CHARACTER_THRESHOLD]
    if has_char.size < system.ncas:                 # fall back to a pure population ranking
        log_msg = (f"only {has_char.size} orbitals exceed the {L_CHARACTER_THRESHOLD} "
                   f"{system.active_l}-character threshold for {system.key}; "
                   f"falling back to a population ranking")
        print(f"  [warn] {log_msg}", flush=True)
        chosen = cand[np.argsort(pop[cand])[-system.ncas:]]
    else:
        chosen = has_char[np.argsort(np.asarray(mo_energy)[has_char])[:system.ncas]]
    return sorted(int(i) for i in chosen)


def make_mf(mol, sfx2c: bool = True):
    from pyscf import scf
    mf = scf.RHF(mol) if mol.spin == 0 else scf.ROHF(mol)
    if sfx2c:
        mf = mf.sfx2c1e()
    mf.conv_tol = SCF_CONV_TOL
    mf.max_cycle = 300
    return mf


def make_mc(mf, system: sysdef.System, nroots: Optional[Dict[int, int]] = None):
    """CASSCF object with the state-averaging protocol of ``system.nroots``.

    A single **common orbital set** is used for all multiplicities (``state_average_mix_``).
    That is deliberate: it mirrors what Kuiva does, where SOC makes spin multiplicity not a
    good quantum number at all and one spinor orbital set must serve the whole manifold. Note
    that OpenMolcas (Tier 2) instead optimises *separate* orbitals per multiplicity, so for
    the two-multiplicity systems (``bi``, ``tlh``) the two tiers are not expected to agree
    beyond gross structure — see ``tests/README.md``.

    ``nroots`` overrides the system's own protocol, which is what
    :func:`scalar_crosscheck_record` uses to build the second, smaller ensemble the Tier-1
    *scalar* comparison needs (``systems.System.scalar_nroots``).
    """
    from pyscf import fci, mcscf
    mc = mcscf.CASSCF(mf, system.ncas, system.nelecas)
    mc.conv_tol = MCSCF_CONV_TOL
    mc.max_cycle_macro = 200
    nroots = dict(system.nroots if nroots is None else nroots)
    total = sum(nroots.values())
    if total == 1 and list(nroots) == [mf.mol.spin + 1]:
        return mc                                   # plain state-specific CASSCF
    if len(nroots) == 1:
        mc.fcisolver.nroots = list(nroots.values())[0]
        return mc.state_average_([1.0 / total] * total)
    solvers = []
    weights: List[float] = []
    for mult, n in sorted(nroots.items()):
        solver = fci.direct_spin1.FCI(mf.mol)
        solver.spin = mult - 1
        solver.nroots = n
        solver = fci.addons.fix_spin(solver, ss=(mult - 1) / 2.0 * ((mult - 1) / 2.0 + 1.0))
        solvers.append(solver)
        weights += [1.0 / total] * n
    return mcscf.state_average_mix_(mc, solvers, weights)


#: Number of roots per multiplicity that SC-NEVPT2 is evaluated for. More than one, because
#: within a degenerate manifold the corrections must come out *equal* — a cheap and rather
#: searching test of a perturbation-theory implementation (it exercises the 4-RDM and the
#: Dyall-Hamiltonian splitting, where a subtle state-dependence bug would show up at once).
NEVPT2_ROOTS_PER_MULT = 3


def run_nevpt2(mf, mc, system: sysdef.System) -> Dict[str, List]:
    """SC-NEVPT2 correlation energies on the state-averaged reference.

    PySCF refuses to run NEVPT2 directly on a state-averaged solver, and rightly so: the
    perturbation must be built for *one* state at a time. The documented recipe is followed
    here — freeze the state-averaged orbitals, redo a multi-root CASCI in them, and correct
    each root separately. Kuiva's own SC-NEVPT2 will have to do exactly the same thing.
    """
    from pyscf import fci, mcscf, mrpt
    out: Dict[str, List] = {}
    for mult, n in sorted(system.nroots.items()):
        nroots = min(NEVPT2_ROOTS_PER_MULT, n)
        # The (na, nb) split must be set explicitly: CASCI infers it from mol.spin, which is
        # only correct for the multiplicity the SCF was run at.
        na = (system.nelecas + mult - 1) // 2
        mc2 = mcscf.CASCI(mf, system.ncas, (na, system.nelecas - na))
        if mult - 1 != mf.mol.spin or n > 1:
            solver = fci.direct_spin1.FCI(mf.mol)
            solver.spin = mult - 1
            s = (mult - 1) / 2.0
            mc2.fcisolver = fci.addons.fix_spin(solver, ss=s * (s + 1.0))
        mc2.fcisolver.nroots = nroots
        mc2.kernel(mc.mo_coeff)
        e_casci = [float(x) for x in np.atleast_1d(mc2.e_tot)][:nroots]
        corr = [float(mrpt.NEVPT(mc2, root=i).kernel()) for i in range(nroots)]
        out[str(mult)] = [{"root": i, "e_casci": e_casci[i], "e_corr": corr[i],
                           "e_total": e_casci[i] + corr[i]} for i in range(nroots)]
    return out


def scalar_crosscheck_record(mf, system: sysdef.System, active: List[int]) -> Optional[Dict]:
    """A second SA-CASSCF over ``system.scalar_nroots``, for the Tier-1 **scalar** comparison.

    ⚠ **Why a second reference exists at all.** ``test_tier1_pyscf`` compares Kuiva's spinor
    CASSCF with SOC off against this file, and calls it an equality rather than an
    approximation. That is only true when both codes average the *same* states, and the two
    define their averages differently: PySCF's is spin adapted, so it averages ``n`` roots of
    one multiplicity and never sees the others, while Kuiva's is "the lowest N spinor roots"
    of a spectrum containing **every** multiplicity. They coincide only while the terms in
    ``nroots`` are the lowest states of the CAS outright.

    ``dy3p`` is where that fails, and it fails in both possible ways at once: quartet terms lie
    below its ⁶P (25683 against 25914 cm^-1 at the reference orbitals), so no count of lowest
    spinor roots is the 21-sextet ensemble; and the counts one would reach for anyway — 126 and
    134 — both land *inside* a 4-fold quartet block, which breaks the state-averaged density's
    invariance and with it Kramers degeneracy. ``scalar_nroots={6: 11}`` takes the ⁶H
    term alone: complete, 8093 cm^-1 clear of anything else, and identical on both sides.

    Returns ``None`` when ``nroots`` already serves, which is every other system.
    """
    from pyscf import mcscf

    if system.scalar_nroots is None or system.scalar_ensemble == dict(system.nroots):
        return None
    t0 = time.time()
    mc = make_mc(mf, system, nroots=system.scalar_ensemble)
    mo = mcscf.sort_mo(mc, mf.mo_coeff, [i + 1 for i in active])       # sort_mo is 1-based
    rec: Dict = {"nroots": {str(k): v for k, v in system.scalar_ensemble.items()},
                 "n_spinor_states": system.scalar_states}
    try:
        rec["e_casscf_avg"] = float(mc.kernel(mo)[0])
    except Exception as exc:                                            # noqa: BLE001
        return {**rec, "error": "CASSCF failed: {}: {}".format(type(exc).__name__, exc)}
    rec["casscf_converged"] = bool(mc.converged)
    es = state_energies(mc)
    rec["e_states"] = es
    rec["e_states_rel_cm"] = [round((e - min(es)) * HARTREE_TO_CM, 4) for e in es]
    rec["seconds"] = round(time.time() - t0, 2)
    return rec


def state_energies(mc) -> List[float]:
    e = getattr(mc, "e_states", None)
    if e is None:
        e = getattr(mc, "e_tot", None)
    return [float(x) for x in np.atleast_1d(e)]


# --- one system -------------------------------------------------------------------------
def run_system(system: sysdef.System, basis_name: str, do_nevpt2: bool = True) -> Dict:
    """Run the full scalar protocol for one system/basis and return a JSON-ready record."""
    from pyscf import mrpt
    t0 = time.time()
    rec: Dict = {"key": system.key, "label": system.label, "basis": basis_name,
                 "charge": system.charge, "spin": system.spin,
                 "ncas": system.ncas, "nelecas": system.nelecas,
                 "active_l": system.active_l,
                 "nroots": {str(k): v for k, v in system.nroots.items()}}
    mol = build_mol(system, basis_name)
    rec["nao"] = int(mol.nao)
    rec["nelectron"] = int(mol.nelectron)

    mf = make_mf(mol)
    e_scf = mf.kernel()
    rec["scf_second_order"] = False
    if not mf.converged:
        # Second-order fallback. A closed-shell RHF for two well-separated open-shell
        # centres coupled to a singlet (ti2cl6_far) is a diradical: the frontier orbitals
        # are near-degenerate and first-order DIIS oscillates rather than converging. This
        # is a property of the reference, not of the molecule - CASSCF fixes it - so the
        # SCF only has to reach a stationary point to hand over sane starting orbitals.
        #
        # Newton/SOSCF: Chaban, Schmidt and Gordon, Theor. Chem. Acc. 97, 88 (1997);
        # PySCF's co-iterative augmented-Hessian implementation follows Sun,
        # J. Chem. Phys. 144, 214103 (2016).
        mf_soscf = mf.newton()
        mf_soscf.conv_tol = SCF_CONV_TOL
        mf_soscf.max_cycle = 300
        e_scf = mf_soscf.kernel(mf.mo_coeff, mf.mo_occ)
        if mf_soscf.converged:
            mf = mf_soscf
            rec["scf_second_order"] = True
    rec["e_scf"] = float(e_scf)
    rec["scf_converged"] = bool(mf.converged)
    rec["e_nuc"] = float(mol.energy_nuc())
    if not mf.converged:
        rec["error"] = "SCF not converged (first- and second-order)"
        return rec

    active = select_active(mol, mf.mo_coeff, mf.mo_energy, system)
    rec["active_mo_indices"] = active
    if system.active_l:
        pop = lowdin_l_populations(mol, mf.mo_coeff, system.active_l)
        rec["active_l_pop"] = [round(float(pop[i]), 6) for i in active]

    from pyscf import mcscf
    mc = make_mc(mf, system)
    mo = mcscf.sort_mo(mc, mf.mo_coeff, [i + 1 for i in active])   # sort_mo is 1-based
    try:
        e_avg = mc.kernel(mo)[0]
    except Exception as exc:                                        # noqa: BLE001
        rec["error"] = f"CASSCF failed: {type(exc).__name__}: {exc}"
        return rec
    rec["casscf_converged"] = bool(mc.converged)
    rec["e_casscf_avg"] = float(e_avg)
    es = state_energies(mc)
    rec["e_states"] = es
    e0 = min(es)
    rec["e_states_rel_cm"] = [round((e - e0) * HARTREE_TO_CM, 4) for e in es]

    scalar = scalar_crosscheck_record(mf, system, active)
    if scalar is not None:
        rec["scalar_crosscheck"] = scalar

    if do_nevpt2:
        t1 = time.time()
        try:
            rec["nevpt2"] = run_nevpt2(mf, mc, system)
            rec["nevpt2_seconds"] = round(time.time() - t1, 2)
        except Exception as exc:                                    # noqa: BLE001
            rec["nevpt2_error"] = f"{type(exc).__name__}: {exc}"

    rec["seconds"] = round(time.time() - t0, 2)
    return rec


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast-only", action="store_true",
                    help="skip systems marked slow (the Ce chlorides)")
    ap.add_argument("--only", default="", help="comma-separated system keys")
    ap.add_argument("--merge", action="store_true",
                    help="merge into the existing reference file instead of replacing it")
    args = ap.parse_args(argv)

    import pyscf
    out: Dict = {"schema": 2, "generator": "tests/generate/tier1_pyscf.py",
                 "pyscf_version": pyscf.__version__,
                 "scf_conv_tol": SCF_CONV_TOL, "mcscf_conv_tol": MCSCF_CONV_TOL,
                 "environment": thermal.describe_environment(),
                 "records": {}}
    if args.merge and REF_OUT.is_file():
        out = json.loads(REF_OUT.read_text())
        out.setdefault("records", {})

    keys = [k for k in args.only.split(",") if k] or None
    for system in sysdef.SYSTEMS:
        if keys and system.key not in keys:
            continue
        if not system.tier1:
            # a derived scalar counterpart (systems.System.tier1) — nothing to generate
            continue
        if args.fast_only and system.slow:
            continue
        # NEVPT2 only where the 4-RDM and the transformation stay laptop-cheap.
        do_nevpt2 = not system.slow and system.ncas <= 7
        for basis_name in dict.fromkeys([system.basis, system.basis_matched]):
            tag = f"{system.key}/{basis_name}"
            with thermal.track_resources() as res:
                try:
                    rec = run_system(system, basis_name, do_nevpt2=do_nevpt2)
                except Exception as exc:                            # noqa: BLE001
                    rec = {"key": system.key, "basis": basis_name,
                           "error": f"{type(exc).__name__}: {exc}"}
            rec["resources"] = res.as_dict()
            out["records"][tag] = rec
            status = rec.get("error", "ok")
            print(f"[tier1] {tag:34s} E_scf={rec.get('e_scf', float('nan')):.8f} "
                  f"E_cas={rec.get('e_casscf_avg', float('nan')):.8f} "
                  f"({res.summary()}) {status}", flush=True)
            if res.throttled:
                # WARNING: the run completed, but its wall time is thermal, not
                # computational - do not read it as a cost estimate.
                print(f"  [warn] {tag}: CPU thermally clamped for "
                      f"{100 * (res.throttle_fraction or 0):.0f}% of the run; the wall time "
                      f"above reflects cooling, not the cost of the calculation", flush=True)

    REF_OUT.parent.mkdir(parents=True, exist_ok=True)
    REF_OUT.write_text(json.dumps(out, indent=2, sort_keys=True))
    n_err = sum(1 for r in out["records"].values() if "error" in r)
    print(f"\nwrote {REF_OUT.relative_to(REPO)}  ({len(out['records'])} records, "
          f"{n_err} with errors)")
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())
