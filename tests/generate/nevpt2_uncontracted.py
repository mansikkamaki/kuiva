"""The uncontracted Dyall-H0 reference for SC-NEVPT2, **with spin-orbit coupling on**.

What this is for
------------------------------------------------------
No external program computes SC-NEVPT2 on a CI-level-SOC two-component reference, so the
scalar-limit PySCF comparison (``tests/test_nevpt2.py``) validates everything **except** the
part that is Kuiva's own: complex integrals with 4-fold symmetry only, a complex CI vector, and
Kramers-degenerate external labels. This generator closes that gap the only way it can be
closed — by solving the same perturbation problem **by brute force over an explicit Fock
space** (``tests/fockspace.py``: dense ladder operators, ``H_D`` assembled as a matrix, class
perturbers as literal projections ``P_l H|Psi_0>``), on integrals that came through the real
front-end with ``with_soc=True``.

⚠ **The reference shares no code with** ``kuiva.pt``, ``kuiva.rdm`` **or** ``kuiva.ci``. That is
the whole point of it and it is why the tool lives here rather than being importable from the
package. It *does* share the class **definition** — the hole/particle counts — and
the shared-implementation rule (a check whose two sides share an implementation cannot see an error in it) one level up applies: a check that shares a definition with its subject cannot fail
on the definition. What caught the one defect of that kind (the contraction group) was a
*third* implementation, PySCF's, in the scalar limit; keep that comparison alive.

How an eight-spinor molecular problem is obtained, and why it is legitimate
---------------------------------------------------------------------------
The Fock space of ``n`` spinors has ``2**n`` states and ``n**2`` dense excitation matrices, so
the reference refuses beyond 8 spinors (``fockspace.MAX_MODES``) — and no all-electron molecule
with a heavy atom has eight spinors. The truncation is therefore **the frozen-core and
deleted-virtual option itself**, which both sides apply:

* Kuiva runs the full molecule and passes ``frozen_core`` / ``deleted_virtual`` thresholds, so
  the frozen spinors keep their mean field in ``F^I`` and ``e_core`` and only vanish from the
  class label ranges.
* The reference is handed exactly that effective problem: the inactive Fock built from the
  **frozen** spinors alone, restricted to the eight retained ones, plus their two-electron
  integrals. Its own inactive space is then the two *correlated* core spinors.

⚠ So this run validates **two** things at once and neither is a proxy for the other: the SOC-on
class algebra, and the claim that the strongly contracted approximation is what the code implements. A frozen-core
implementation that projected the frozen orbitals out of ``H`` instead of out of the *labels*
would agree with nothing here.

The system
----------
``systems.hi`` — HI at its experimental ``r_e`` in the project default basis, sigma/sigma* CAS,
SOC ingested and **screening off** (the correction changes no scalar quantity and costs a
four-component atomic solve for iodine; what is under test is the perturbation algebra, not the
Hamiltonian). The retained window is the highest occupied Kramers pair, the four active
spinors, and the lowest virtual Kramers pair. Iodine's 5p spin-orbit splitting is thousands of
cm^-1, so the retained integrals are genuinely complex — which is what makes this a SOC-on
check rather than a complex-arithmetic formality. The measured imaginary weight is recorded in
the file so that a later reader can see it was not a rounding-level effect.

Cost: seconds. Usage::

    python tests/generate/nevpt2_uncontracted.py            # regenerate the committed file
    python tests/generate/nevpt2_uncontracted.py --check    # recompute and compare, write nothing
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "tests"))

import systems as sysdef                                                    # noqa: E402
from fockspace import MAX_MODES, ReferenceNEVPT2                            # noqa: E402

REF_OUT = REPO / "tests/reference/nevpt2_uncontracted.json"

#: Schema of the emitted file. Bump when the *meaning* of a stored field changes.
SCHEMA = 1

#: Working-memory limit for the generator [GB].
MEMORY_GB = 6.0

#: Spinors retained on each side of the active space. Two and two, because the Fock-space
#: reference refuses beyond eight modes in total and the active space is four.
N_CORE_KEPT = 2
N_VIRTUAL_KEPT = 2


def _group_edges(eps: np.ndarray, rtol: float = 1e-9) -> List[int]:
    """Indices at which a new exactly-degenerate group starts in an ascending ``eps``."""
    scale = max(float(np.max(np.abs(eps))), 1.0) if eps.size else 1.0
    return [0] + [i + 1 for i in range(eps.size - 1) if eps[i + 1] - eps[i] > rtol * scale]


def thresholds(eps_inactive: np.ndarray, eps_virtual: np.ndarray):
    """Frozen-core / deleted-virtual thresholds keeping exactly the requested window.

    ⚠ **Derived from the degeneracy groups, not from a count**, and it refuses if the group
    boundary does not land where the window needs it: cutting a degenerate group in half is
    what the group rule forbids, and a generator that quietly took "the two highest" would be producing a
    reference for a calculation Kuiva refuses to run.
    """
    starts_i = _group_edges(eps_inactive)
    starts_v = _group_edges(eps_virtual)
    cut_i = eps_inactive.size - N_CORE_KEPT
    cut_v = N_VIRTUAL_KEPT
    if cut_i not in starts_i:
        raise SystemExit("the {} highest inactive spinors are not a whole set of degenerate "
                         "groups; eps tail {}".format(N_CORE_KEPT,
                                                      np.round(eps_inactive[-4:], 6)))
    if cut_v not in starts_v:
        raise SystemExit("the {} lowest virtual spinors are not a whole set of degenerate "
                         "groups; eps head {}".format(N_VIRTUAL_KEPT,
                                                      np.round(eps_virtual[:4], 6)))
    frozen = 0.5 * float(eps_inactive[cut_i - 1] + eps_inactive[cut_i])
    deleted = 0.5 * float(eps_virtual[cut_v - 1] + eps_virtual[cut_v])
    return frozen, deleted


def build(system_key: str = "hi", *, memory_gb: float = MEMORY_GB) -> Dict:
    """Run both sides and return the record."""
    from kuiva.interface import api
    from kuiva.mcscf.orbopt import CASIntegrals, OrbitalSpaces
    from kuiva.pt.classes import available_classes
    from kuiva.pt.nevpt2 import pseudo_canonicalize, sc_nevpt2
    from kuiva.util import resources as res

    res.clear()
    system = sysdef.get(system_key)
    t0, c0 = time.time(), time.process_time()

    molecule = api.Molecule(atoms=system.atoms, basis=system.basis, charge=system.charge,
                            spin=system.spin)
    reference = api.spinor_reference(molecule, memory_gb=memory_gb, screening="none")
    coeff = reference.spinors_in_ao()
    h_ao = reference.h_one_electron()
    n_active = 2 * system.ncas
    n_inactive = reference.data.nelec_total - system.nelecas
    if n_inactive % 2:
        raise SystemExit("an odd inactive spinor count would split a Kramers pair")
    spaces = OrbitalSpaces.from_counts(n_inactive, n_active, reference.nspinor)

    cas = api.casci(reference, active=list(range(n_inactive, n_inactive + n_active)),
                    n_active_elec=system.nelecas, n_states=2, report=False)
    gap_cm = float((cas.energies[1] - cas.energies[0]) * 219474.6313632)
    if gap_cm < 100.0:
        raise SystemExit("the CAS ground state is degenerate to {:.3f} cm^-1, so the two codes "
                         "may pick different mixtures of the block and the comparison would be "
                         "meaningless".format(gap_cm))

    # ⚠ **The ground state's own 1-RDM, not the two-state average `cas.gamma`.** The driver
    # below canonicalizes with the density of the state it is correcting, and a Fock built from
    # a different density gives different `eps` — which showed up here as a 0.3% disagreement
    # in ``Sijrs``, the one class that contains no active-space quantity at all and therefore
    # could only be reporting an orbital-energy difference.
    from kuiva.ci.strings import CASSpace
    from kuiva.rdm.rdm import RDMBuilder
    gamma0 = RDMBuilder(CASSpace(n_active, system.nelecas))(
        cas.vectors[0], enforce_kramers=False)[0]
    canonical = pseudo_canonicalize(reference.factors, h_ao, coeff, spaces, gamma0,
                                    e_nuc=reference.data.e_nuc)
    frozen_core, deleted_virtual = thresholds(canonical.eps_inactive, canonical.eps_virtual)

    # --- Kuiva's answer, through the production driver ------------------------------------
    corrected = sc_nevpt2(reference.factors, h_ao, coeff, spaces,
                          cas.vectors[0], system.nelecas,
                          energies=[float(cas.energies[0])], e_nuc=reference.data.e_nuc,
                          frozen_core=frozen_core, deleted_virtual=deleted_virtual,
                          report=False)

    # --- the same effective problem, by brute force ----------------------------------------
    core_kept = spaces.inactive[-N_CORE_KEPT:]
    virt_kept = spaces.virtual[:N_VIRTUAL_KEPT]
    retained = np.concatenate([core_kept, spaces.active, virt_kept])
    frozen_idx = spaces.inactive[:-N_CORE_KEPT]
    dropped = spaces.virtual[N_VIRTUAL_KEPT:]
    if retained.size > MAX_MODES:
        raise SystemExit("{} retained spinors exceeds the brute-force limit of {}"
                         .format(retained.size, MAX_MODES))
    # ⚠ A *fake* partition whose "inactive" is exactly the frozen set: what comes back is the
    # inactive Fock and core energy of the frozen orbitals alone, which is precisely the
    # effective one-electron problem a frozen core leaves behind. Nothing here is a calculation Kuiva
    # runs; it is a way of extracting integrals.
    effective = CASIntegrals.build(reference.factors, h_ao, canonical.coeff,
                                   OrbitalSpaces(inactive=frozen_idx, active=retained,
                                                 virtual=dropped, n_orb=spaces.n_orb),
                                   e_nuc=reference.data.e_nuc)
    h_eff = np.ascontiguousarray(effective.f_inactive[np.ix_(retained, retained)])
    eri_eff = effective.active_eri()
    imaginary_weight = float(np.max(np.abs(eri_eff.imag)) / np.max(np.abs(eri_eff.real)))

    n_core, n_act = int(core_kept.size), int(spaces.n_active)
    brute = ReferenceNEVPT2(
        h_eff, eri_eff,
        inactive=list(range(n_core)),
        active=list(range(n_core, n_core + n_act)),
        virtual=list(range(n_core + n_act, retained.size)),
        eps_inactive=canonical.eps_inactive[-N_CORE_KEPT:],
        eps_virtual=canonical.eps_virtual[:N_VIRTUAL_KEPT],
        n_active_elec=system.nelecas)
    residual = brute.dyall_residual(0)

    record: Dict = {
        "schema": SCHEMA, "key": system.key, "label": system.label,
        "basis": system.basis, "ncas": system.ncas, "nelecas": system.nelecas,
        "geometry": system.geom_note, "screening": "none", "with_soc": True,
        "hamiltonian": (reference.data.soc.provenance()
                        if reference.data.soc is not None else None),
        "nao": int(reference.data.nao), "nspinor": int(reference.nspinor),
        "cholesky_tol": reference.factors.tol,
        "cholesky_naux": int(reference.factors.naux),
        "cholesky_orbit_complete": bool(reference.factors.orbit_complete),
        "n_retained_spinors": int(retained.size),
        "n_frozen": int(corrected.n_frozen), "n_deleted": int(corrected.n_deleted),
        "frozen_core_threshold": frozen_core, "deleted_virtual_threshold": deleted_virtual,
        "eps_inactive_kept": [float(e) for e in canonical.eps_inactive[-N_CORE_KEPT:]],
        "eps_virtual_kept": [float(e) for e in canonical.eps_virtual[:N_VIRTUAL_KEPT]],
        "cas_gap_cm": gap_cm,
        "eri_imaginary_over_real": imaginary_weight,
        "dyall_residual": residual,
        "e_casscf": float(corrected.e_casscf[0]),
        "e2_kuiva": float(corrected.e2[0]),
        "classes": {},
    }
    e2_reference = 0.0
    for name in available_classes():
        norm, energy = brute.by_name(name)
        uncontracted = brute.class_energy_uncontracted(*brute.CLASS_PATTERN[name])
        e2_reference += energy
        record["classes"][name] = {
            "norm": norm, "energy": energy, "energy_uncontracted": uncontracted,
            "norm_kuiva": float(corrected.class_norms[name][0]),
            "energy_kuiva": float(corrected.class_energies[name][0]),
        }
    record["e2_reference"] = e2_reference
    record["wall_seconds"] = time.time() - t0
    record["cpu_seconds"] = time.process_time() - c0
    return record


def report(record: Dict) -> float:
    """Print the comparison and return the worst relative class deviation."""
    print("{}  {} spinors retained ({} frozen, {} deleted); "
          "max|Im(pq|rs)| / max|Re| = {:.2e}".format(
              record["label"], record["n_retained_spinors"], record["n_frozen"],
              record["n_deleted"], record["eri_imaginary_over_real"]))
    print("  ||(H_D - E_0)|Psi_0>|| = {:.2e}   (the reference's own consistency check)"
          .format(record["dyall_residual"]))
    print("  {:8s} {:>22s} {:>22s} {:>10s} {:>16s}".format(
        "class", "brute force E", "kuiva E", "rel", "uncontracted"))
    worst = 0.0
    for name, entry in record["classes"].items():
        rel = (abs(entry["energy_kuiva"] - entry["energy"])
               / max(abs(entry["energy"]), 1e-30))
        worst = max(worst, rel)
        print("  {:8s} {: .15e} {: .15e} {:9.2e} {: .9e}".format(
            name, entry["energy"], entry["energy_kuiva"], rel,
            entry["energy_uncontracted"]))
    print("  E2 total  {: .15e} (brute force)  {: .15e} (kuiva)".format(
        record["e2_reference"], record["e2_kuiva"]))
    print("  worst relative class deviation {:.2e}; {:.1f} s wall / {:.1f} s cpu".format(
        worst, record["wall_seconds"], record["cpu_seconds"]))
    return worst


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--system", default="hi")
    parser.add_argument("--memory-gb", type=float, default=MEMORY_GB)
    parser.add_argument("--check", action="store_true",
                        help="recompute and compare against the committed file; write nothing")
    args = parser.parse_args(argv)

    record = build(args.system, memory_gb=args.memory_gb)
    worst = report(record)
    if args.check:
        stored = json.loads(REF_OUT.read_text())
        for name, entry in record["classes"].items():
            old = stored["classes"][name]["energy"]
            print("  {:8s} stored {: .15e}  now {: .15e}  d={:.2e}".format(
                name, old, entry["energy"], abs(old - entry["energy"])))
        return 0
    REF_OUT.parent.mkdir(parents=True, exist_ok=True)
    REF_OUT.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    print("wrote {}".format(REF_OUT))
    return 0 if worst < 1e-9 else 1


if __name__ == "__main__":
    raise SystemExit(main())
