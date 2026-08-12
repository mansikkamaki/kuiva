"""Characterize the DLU approximation on molecules.

**What this is for, and what it is not.** DLU is the bottom rung of the decoupling cost ladder and
X2C-AMF is the default everywhere (a user decision). So this script does *not* decide a
default — it measures what a user buys when they reach for the escape hatch, and the numbers
are recorded in the local validation notes.

⚠ **The reference is ``partition="single"``, never the ``"1e"`` route.** Kuiva's exact
decoupling and PySCF's differ by up to 2.4e-07 relative on a heavy element, because Kuiva
projects linear dependence out of the four-component metric and PySCF does not. That
difference is the same size as the effect being measured, so comparing DLU against PySCF would
confound the two. ``partition="single"`` is the exact transformation through identical code.

Four measurements, in increasing distance from a matrix norm:

1. **Operator norms** — ``max |h_DLU - h_exact| / max |h|``, split into the spin-free and
   spin-orbit parts, plus ``max |X|`` per fragment and the off-diagonal weight of the
   four-component Hamiltonian (the cheap predictor of the DLU error).
2. **First-order energy shift** — ``Tr[(h_DLU - h_exact) D]`` over the converged scalar SCF
   density, lifted to two components. This is the leading error in a total energy, in Eh, which
   is what a user actually feels. ⚠ A spin-free density is blind to the spin-orbit part of the
   difference, which is why (3) exists.
3. **One-electron spectrum** — the generalized eigenproblem ``h C = S C e`` in the molecular AO
   basis, no SCF. ⚠ **State the construction whenever quoting a splitting**: this is a
   frozen-basis one-electron spectrum, not a correlated or self-consistent one, and the two
   disagree on absolute splittings by ~30% while agreeing on what an approximation does to them.
4. **A geometry scan on TlH** — the decisive one. A constant energy error is harmless; an error
   that *varies with geometry* is a spurious force. Reported as dE/dR in Eh/bohr against the
   1e-4 Eh/bohr that a geometry optimization typically converges to.

Run:  ``python tests/generate/dlu_accuracy.py [--out temp/dlu_accuracy.json]``
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import scipy.linalg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from kuiva.interface.pyscf_bridge import (four_component_one_electron,  # noqa: E402
                                          ingest_spin_orbit, local_x2c_hamiltonian,
                                          molecular_partition, run_scalar_x2c)
from kuiva.spinor.expand import decompose_two_component, spin_block_diagonal  # noqa: E402
from kuiva.util import resources as res                                      # noqa: E402
from kuiva.x2c.local import off_block_weight                                  # noqa: E402

#: ⚠ Hard wall budget. The run is designed to finish in ~5 minutes; this exists so a
#: pathological case terminates rather than being waited on, and every result already computed
#: is on disk by then because the file is rewritten after each system.
WALL_BUDGET_S = 9 * 60

#: Systems, cheapest first, so a killed run still yields the cheap ones. Geometries come from
#: the single source of truth so that a Tier-1/Tier-2 counterpart exists for each.
KEYS = ("tlh", "ticl3", "cecl3", "ti2cl6")


def _system(key):
    from tests.generate import systems as S
    return [s for s in S.SYSTEMS if s.key == key][0]


def _mole(system, scale=None):
    from pyscf import gto
    atoms = [(a, tuple(c)) for a, c in system.atoms]
    if scale is not None:                    # only used by the diatomic scan
        atoms = [(a, tuple(x * scale for x in c)) for a, c in atoms]
    return gto.M(atom=atoms, basis=system.basis, charge=system.charge, spin=system.spin,
                 verbose=0)


def _relative(a, b):
    return float(np.max(np.abs(a - b))) / float(np.max(np.abs(b)))


def _one_electron_spectrum(h2c, s_ao, n_occupied):
    """Eigenvalues of ``h C = S C e`` in the AO basis — construction (3) of the docstring."""
    s2c = spin_block_diagonal(np.asarray(s_ao))
    values = scipy.linalg.eigh(np.asarray(h2c), s2c, eigvals_only=True)
    return np.asarray(values[:n_occupied])


def measure_system(key, verbose=True, memory_gb=8.0):
    """Every measurement for one system. Returns a plain dict (JSON-serializable)."""
    system = _system(key)
    mol = _mole(system)
    record = {"key": key, "label": system.label, "basis": system.basis,
              "nao": int(mol.nao), "natm": int(mol.natm),
              "charge": int(mol.charge), "spin": int(mol.spin)}

    # --- 1. Operator norms, and the cost of each route -----------------------------------
    blocks = four_component_one_electron(mol)
    record["nao_working"] = int(blocks.nao)
    partition = molecular_partition(mol, blocks)
    record["off_block_weight"] = off_block_weight(blocks.hcore.ll, partition)

    timings = {}
    hamiltonians = {}
    for name, kwargs in (("exact", dict(partition="single")),
                         ("dlu_diagonal", dict(partition="atoms", source="diagonal")),
                         ("dlu_isolated", dict(partition="atoms", source="isolated"))):
        t0, c0 = time.time(), time.process_time()
        h, dlu_record = local_x2c_hamiltonian(mol, **kwargs)
        timings[name] = {"wall_s": time.time() - t0, "cpu_s": time.process_time() - c0}
        hamiltonians[name] = h
        if name == "dlu_diagonal":
            record["block_scales"] = dict(dlu_record.block_scales)
    # PySCF's exact route, for the confound the reference exists to avoid.
    t0, c0 = time.time(), time.process_time()
    h_pyscf = ingest_spin_orbit(mol, approx="1e", screening="none").hamiltonian()
    timings["pyscf_1e"] = {"wall_s": time.time() - t0, "cpu_s": time.process_time() - c0}
    record["timings"] = timings
    record["speedup_cpu_dlu_vs_exact"] = (timings["exact"]["cpu_s"]
                                          / max(timings["dlu_diagonal"]["cpu_s"], 1e-9))

    exact = hamiltonians["exact"]
    sf_exact, w_exact = decompose_two_component(exact)
    errors = {}
    for name in ("dlu_diagonal", "dlu_isolated", "pyscf_1e"):
        other = h_pyscf if name == "pyscf_1e" else hamiltonians[name]
        sf, w = decompose_two_component(other)
        errors[name] = {
            "relative": _relative(other, exact),
            "spin_free_abs_eh": float(np.max(np.abs(sf - sf_exact))),
            "spin_orbit_abs_eh": float(np.max(np.abs(w - w_exact))),
            "spin_orbit_relative": (float(np.max(np.abs(w - w_exact)))
                                    / float(np.max(np.abs(w_exact)))),
        }
    record["operator_error"] = errors
    record["soc_scale_eh"] = float(np.max(np.abs(w_exact)))

    # --- 2 & 3. Energy and spectrum ------------------------------------------------------
    # ⚠ The SCF may be refused by the resource pre-flight before it starts — Ti2Cl6's spinor MO
    # transform needs ~17 GB, which this machine does not have. That refusal is correct and
    # must not be worked around by inflating the limit to get a number. The operator-level
    # measurements above need no density, so the system still contributes them; what is missing
    # is recorded as missing rather than silently absent.
    t0, c0 = time.time(), time.process_time()
    try:
        data = run_scalar_x2c(system_molecule(system), screening="none", memory_gb=memory_gb)
    except MemoryError as exc:
        record["energy_shift"] = None
        record["energy_shift_skipped"] = str(exc).splitlines()[0]
        if verbose:
            e = errors["dlu_diagonal"]
            print("  {:8s} nao={:3d}/{:4d}  rel={:.2e}  dh_sf={:.2e} Eh  dw={:.2e} Eh  "
                  "dE1=(no SCF: memory)  speedup={:.1f}x".format(
                      key, record["nao"], record["nao_working"], e["relative"],
                      e["spin_free_abs_eh"], e["spin_orbit_abs_eh"],
                      record["speedup_cpu_dlu_vs_exact"]), flush=True)
        return record
    record["timings"]["scalar_scf"] = {"wall_s": time.time() - t0,
                                       "cpu_s": time.process_time() - c0}
    record["scf_energy_eh"] = float(data.e_scf)

    # Scalar density lifted to two components: D_2c = diag(D/2, D/2), so Tr[h_2c D_2c] is the
    # one-electron energy of the same state under either Hamiltonian.
    dm = _scalar_density(data, mol.nao)
    dm_2c = spin_block_diagonal(0.5 * dm)

    n_occupied = int(round(dm.trace()))
    shifts = {}
    spectra = {"exact": _one_electron_spectrum(exact, data.s_ao, n_occupied)}
    for name in ("dlu_diagonal", "dlu_isolated"):
        difference = hamiltonians[name] - exact
        shifts[name] = {
            "first_order_energy_eh": float(np.real(np.einsum("ij,ji->", difference, dm_2c))),
        }
        spectra[name] = _one_electron_spectrum(hamiltonians[name], data.s_ao, n_occupied)
        delta = spectra[name] - spectra["exact"]
        shifts[name]["max_orbital_shift_eh"] = float(np.max(np.abs(delta)))
        shifts[name]["max_orbital_shift_cm"] = float(np.max(np.abs(delta))) * 219474.6313632
        # The frontier gap is the difference of two eigenvalues, so a common shift cancels —
        # which is the point: a spectroscopic quantity is far less sensitive than a total.
        gap_exact = spectra["exact"][-1] - spectra["exact"][0]
        gap_other = spectra[name][-1] - spectra[name][0]
        shifts[name]["occupied_span_shift_cm"] = float(gap_other - gap_exact) * 219474.6313632
    record["energy_shift"] = shifts
    record["n_occupied"] = n_occupied

    if verbose:
        e = errors["dlu_diagonal"]
        print("  {:8s} nao={:3d}/{:4d}  rel={:.2e}  dh_sf={:.2e} Eh  dw={:.2e} Eh  "
              "dE1={:+.3e} Eh  speedup={:.1f}x".format(
                  key, record["nao"], record["nao_working"], e["relative"],
                  e["spin_free_abs_eh"], e["spin_orbit_abs_eh"],
                  shifts["dlu_diagonal"]["first_order_energy_eh"],
                  record["speedup_cpu_dlu_vs_exact"]), flush=True)
    return record


def _scalar_density(data, nao):
    """Total scalar AO density from an ingested reference, restricted or not.

    ⚠ Via ``mo_sets()``, the only sanctioned way to consume the orbitals: branching on
    ``mo_coeff.ndim`` at a call site is how a ``(2, nao, nmo)`` array eventually gets treated
    as ``(nao, nmo)`` of a different basis. A restricted ``mo_occ`` already carries the full
    occupation (2.0), an unrestricted one carries a set per spin.
    """
    sets = data.mo_sets()
    occupations = data.mo_occ if data.unrestricted else (data.mo_occ,)
    dm = np.zeros((nao, nao))
    for coefficients, occupation in zip(sets, occupations):
        c = np.asarray(coefficients)
        dm = dm + (c * np.asarray(occupation)) @ c.T
    return dm


def system_molecule(system):
    """The :class:`kuiva.interface.Molecule` for a Tier-1 system (the API path)."""
    from kuiva.interface import Molecule
    return Molecule(atoms=[(a, tuple(c)) for a, c in system.atoms], basis=system.basis,
                    charge=system.charge, spin=system.spin)


def scan_tlh(points=(0.92, 0.96, 1.0, 1.04, 1.08), verbose=True):
    """⚠ The decisive measurement: does the DLU error vary with geometry?

    A constant error is absorbed into everything a chemist compares; an error that changes with
    bond length is a **spurious force** and would move an optimized geometry. Reported as the
    slope of the first-order energy error, in Eh/bohr, against the ~1e-4 Eh/bohr that a
    geometry optimization is normally converged to.
    """
    system = _system("tlh")
    r_e = float(np.linalg.norm(np.asarray(system.atoms[1][1]) - np.asarray(system.atoms[0][1])))
    rows = []
    for scale in points:
        mol = _mole(system, scale=scale)
        distance = r_e * scale
        exact, _ = local_x2c_hamiltonian(mol, partition="single")
        local, _ = local_x2c_hamiltonian(mol, partition="atoms")

        # The same first-order measure as above, on a density that is recomputed per geometry.
        from kuiva.interface import Molecule
        atoms = [(a, tuple(x * scale for x in c)) for a, c in system.atoms]
        data = run_scalar_x2c(Molecule(atoms=atoms, basis=system.basis, charge=system.charge,
                                       spin=system.spin), screening="none", memory_gb=8.0)
        dm = _scalar_density(data, mol.nao)
        shift = float(np.real(np.einsum("ij,ji->", local - exact, spin_block_diagonal(0.5 * dm))))
        rows.append({"scale": scale, "r_angstrom": distance,
                     "relative_error": _relative(local, exact),
                     "first_order_energy_eh": shift,
                     "scf_energy_eh": float(data.e_scf)})
        if verbose:
            print("  R = {:.4f} A  rel={:.2e}  dE1={:+.6e} Eh".format(
                distance, rows[-1]["relative_error"], shift), flush=True)

    r_bohr = np.array([row["r_angstrom"] for row in rows]) / 0.52917721092
    shifts = np.array([row["first_order_energy_eh"] for row in rows])
    slope = float(np.polyfit(r_bohr, shifts, 1)[0])
    return {"points": rows, "spurious_force_eh_per_bohr": slope,
            "energy_error_range_eh": float(shifts.max() - shifts.min()),
            "geometry_convergence_threshold_eh_per_bohr": 1e-4}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="temp/dlu_accuracy.json")
    parser.add_argument("--keys", nargs="*", default=list(KEYS))
    parser.add_argument("--scan-only", action="store_true")
    parser.add_argument("--memory-gb", type=float, default=8.0)
    args = parser.parse_args()

    res.ensure_configured(8.0)
    started = time.time()
    results = {"systems": [], "scan": None,
               "note": "reference is partition='single' (Kuiva's exact decoupling through the "
                       "same code), never the PySCF '1e' route - see the module docstring"}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    def flush():
        with open(args.out, "w") as handle:
            json.dump(results, handle, indent=2, sort_keys=True)

    print("DLU accuracy characterization; budget {} s".format(WALL_BUDGET_S))
    for key in ([] if args.scan_only else args.keys):
        if time.time() - started > WALL_BUDGET_S:
            print("  wall budget reached; stopping with {} systems done"
                  .format(len(results["systems"])), flush=True)
            break
        results["systems"].append(measure_system(key, memory_gb=args.memory_gb))
        flush()                                # incremental, per 12.0

    if time.time() - started < WALL_BUDGET_S:
        print("TlH geometry scan:")
        results["scan"] = scan_tlh()
        flush()

    results["wall_s"] = time.time() - started
    flush()
    print("wrote {} ({:.0f} s)".format(args.out, results["wall_s"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
