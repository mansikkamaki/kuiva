"""End-to-end molecular X2CAMF runs.

Runs the front-end twice on each system — ``screening="none"`` and ``screening="x2camf"``
— and records what the two-electron picture change does to a **molecular** Hamiltonian: the
size of each half of the correction, the change in the spin-orbit operator, the number of
four-component atomic solves, and the cost of each.

⚠ **This is not a validation against another program**, and it deliberately does not pretend
to be. The molecular Tier-2 comparison compares *SOC state energies and
moment matrices*, which needed the CI layer before it existed; what can be compared today
is the one-electron operator and, where the plugin is installed, the correction itself against
a second implementation of the same method. Both are recorded here. The number that says the
correction is right is atomic and lives in ``tests/reference/x2camf_dirac.json`` .

⚠ **Cost, and the ten-minute ad-hoc rule.** An atomic solve is per element and steeply
``Z``-dependent, and average-of-configuration multiplies it: Cl is seconds, Ti
about a minute, I about a quarter of an hour, Ce(3+) about forty minutes. The cheap systems
are the default; ``--systems`` opts into the expensive ones explicitly, and every run writes
its record as soon as it finishes so that a killed run still yields what it completed.

Usage::

    python tests/generate/amf_molecular.py                     # ne, ticl3, ti2cl6
    python tests/generate/amf_molecular.py --systems hi cecl3  # the expensive ones
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ⚠ The persistent correction cache is turned off *before* kuiva is imported, and this
# generator must not be run without it. It records a **solve count** — the molecular
# statement that a molecule pays one four-component solve per unique element and not one
# per atom — and a cached element is a hit rather than a solve, so on a machine that
# already has Cl the record reads 1 where a fresh clone reads 2. Both are true and only
# one is a reference. (``disk_hits`` is recorded too, and the test asserts on the sum,
# because a committed record should still be interpretable if this is ever bypassed.)
os.environ.setdefault("KUIVA_AMF_CACHE", "off")

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tests.generate import thermal                                          # noqa: E402
from tests.generate.systems import SYSTEMS_BY_KEY                           # noqa: E402

OUTPUT = REPO / "tests/reference/amf_molecular.json"

#: Systems whose atomic solves are seconds to a couple of minutes, i.e. inside the ten-minute rule.
CHEAP = ("ne", "ticl3", "ti2cl6")

#: ⚠ Where the configured **site** limit (``defaults.conf``, 8 GB on the dev box) is not enough and the calculation is asked for anyway. This is not a workaround: the
#: pre-flight refused ``ti2cl6`` at 8 GB naming ``B^P_pq`` over all 420 spinors as the phase
#: that peaks — a real requirement of that system, unrelated to the correction, whose own
#: arrays are 0.001 GB — and printed the number that would run it. Recorded per system so a
#: reader can see which records were taken above the site default and by how much.
MEMORY_GB = {"ti2cl6": 9.5, "cecl3": 9.5}


def measure(key: str, plugin: bool = True, memory_gb=None) -> dict:
    """One system, both Hamiltonians, plus the plugin cross-check where it is available.

    The Hamiltonians come through :func:`kuiva.interface.api.scalar_x2c_reference`, i.e. the
    whole front-end and not just the correction: the pre-flight, the SCF, the ingestion and
    the standard output block. A correction that assembles correctly but cannot be reached from the
    driver is not wired in.
    """
    from kuiva.amf import amf_correction
    from kuiva.amf import x2camf_plugin
    from kuiva.amf.atomic import cache_statistics, clear_cache
    from kuiva.interface.api import Molecule, scalar_x2c_reference
    from kuiva.interface.pyscf_bridge import build_mole
    from kuiva.util import resources as res

    system = SYSTEMS_BY_KEY[key]
    molecule = Molecule(atoms=[(s, tuple(xyz)) for s, xyz in system.atoms],
                        basis=system.basis, charge=system.charge, spin=system.spin)
    mol = build_mole(molecule)

    clear_cache()
    # ⚠ Each reference is scoped (``in_phase``) because the two are **sequential**, not
    # simultaneous: the unscreened run's ERI array is dropped with its ``ScalarX2CData`` the
    # moment ``.soc`` is taken off it. Without the scope its reservation stays committed and
    # the second run is refused for memory that is not in use — which is what happened the
    # first time this generator was run, and is exactly the pessimism that makes a hard
    # limit unusable.
    with res.in_phase("unscreened reference"):
        plain = scalar_x2c_reference(molecule, memory_gb=memory_gb,
                                     screening="none").soc
    with thermal.track_resources() as t, res.in_phase("screened reference"):
        screened_data = scalar_x2c_reference(molecule, screening="x2camf",
                                             memory_gb=memory_gb)
    screened = screened_data.soc
    correction = amf_correction(mol, method="x2camf")        # from the cache; for the blocks

    slices = mol.aoslice_by_atom()
    per_element = {}
    for ia in range(mol.natm):
        label = mol.atom_symbol(ia)
        p0, p1 = int(slices[ia][2]), int(slices[ia][3])
        per_element.setdefault(label, {
            "n_atoms": 0,
            "nao": p1 - p0,
            "configuration": correction.configurations[label],
            "dh_sf": float(np.max(np.abs(correction.h_sf[p0:p1, p0:p1]))),
            "dw": float(np.max(np.abs(correction.w[:, p0:p1, p0:p1]))),
            "w_one_electron": float(np.max(np.abs(plain.w[:, p0:p1, p0:p1]))),
        })["n_atoms"] += 1

    record = {
        "label": system.label,
        "basis": system.basis,
        "natm": int(mol.natm),
        "nao": int(mol.nao),
        "elements": list(correction.elements),
        "solves": cache_statistics()["solves"],
        # ⚠ Recorded beside ``solves`` because the two together are the invariant that
        # means something: each unique element is *acquired* exactly once, whether by
        # solving or from the persistent cache. ``solves`` alone is a statement about
        # the machine the generator ran on.
        "disk_hits": cache_statistics()["disk_hits"],
        "e_scf": float(screened_data.e_scf),
        "scf_converged": bool(screened_data.converged),
        "memory_gb": memory_gb,
        "off_atom_nonzeros": _off_atom_nonzeros(mol, correction),
        "soc_strength_one_electron": plain.soc_strength,
        "soc_strength_screened": screened.soc_strength,
        "soc_reduction": 1.0 - screened.soc_strength / plain.soc_strength,
        "spin_free_scale": correction.spin_free_scale,
        "spin_orbit_scale": correction.spin_orbit_scale,
        "tr_residual_rel": correction.tr_residual_rel,
        "per_element": per_element,
        "provenance": screened.provenance(),
        "resources": t.as_dict(),
    }

    if plugin and x2camf_plugin.available():
        record["plugin"] = _plugin_comparison(mol, correction)
    return record


def _off_atom_nonzeros(mol, correction) -> int:
    slices = mol.aoslice_by_atom()
    inside = 0
    for ia in range(mol.natm):
        p0, p1 = int(slices[ia][2]), int(slices[ia][3])
        inside += int(np.count_nonzero(correction.h_sf[p0:p1, p0:p1]))
        inside += int(np.count_nonzero(correction.w[:, p0:p1, p0:p1]))
    total = int(np.count_nonzero(correction.h_sf)) + int(np.count_nonzero(correction.w))
    return total - inside


def _plugin_comparison(mol, ours) -> dict:
    """Kuiva's assembly against the plugin's own.

    ⚠ Only meaningful where Kuiva's per-element defaults are the **neutral** atom, which the
    plugin is fixed to; a molecule containing an f-block element takes Kuiva's M(3+) default
    and the two are then not the same calculation. That is reported rather than worked around.
    """
    from kuiva.amf import amf_correction
    from kuiva.amf.configuration import AtomicConfiguration

    labels = {mol.atom_symbol(ia): mol.atom_pure_symbol(ia) for ia in range(mol.natm)}
    mismatched = sorted(
        label for label, pure in labels.items()
        if ours.configurations[label] != AtomicConfiguration.ground(pure).canonical)
    if mismatched:
        return {"comparable": False, "non_neutral_reference": mismatched}

    with thermal.track_resources() as t:
        theirs = amf_correction(mol, method="x2camf-external")
    return {
        "comparable": True,
        "off_atom_nonzeros": _off_atom_nonzeros(mol, theirs),
        "dw_relative": float(np.max(np.abs(ours.w - theirs.w))
                             / (np.max(np.abs(ours.w)) or 1.0)),
        "dh_sf_relative": float(np.max(np.abs(ours.h_sf - theirs.h_sf))
                                / (np.max(np.abs(ours.h_sf)) or 1.0)),
        "resources": t.as_dict(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", nargs="+", default=list(CHEAP),
                        help="system keys from tests/generate/systems.py")
    parser.add_argument("--no-plugin", action="store_true",
                        help="skip the x2camf plugin cross-check even where it is installed")
    args = parser.parse_args()

    stored = json.loads(OUTPUT.read_text()) if OUTPUT.is_file() else {
        "schema": 1, "generator": "tests/generate/amf_molecular.py", "systems": {}}

    for key in args.systems:
        started = time.time()
        print("=== {} ...".format(key), flush=True)
        stored["systems"][key] = measure(key, plugin=not args.no_plugin,
                                         memory_gb=MEMORY_GB.get(key))
        # ⚠ Written after every system, not at the end: a run killed at its budget
        # must still yield the systems it completed.
        OUTPUT.write_text(json.dumps(stored, indent=2, sort_keys=True) + "\n")
        record = stored["systems"][key]
        print("    {} solves, max|dw| = {:.3e} Eh, SOC reduced {:.1%}, {:.0f} s wall / "
              "{:.0f} s cpu".format(record["solves"], record["spin_orbit_scale"],
                                    record["soc_reduction"],
                                    record["resources"]["wall_seconds"],
                                    record["resources"]["cpu_seconds"]), flush=True)
        print("    ({:.0f} s total for this system)".format(time.time() - started),
              flush=True)
    print("wrote {}".format(OUTPUT.relative_to(REPO)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
