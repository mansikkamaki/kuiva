"""Cost and convergence of average-of-configuration atomic DHF.

The four-component atomic solve is cheap only at the light end — roughly ``n4c^4``, i.e. 0.2 s
wall for Ne but 177 s for Xe in the *uncontracted* default basis — and the open-shell ion set
is the expensive end of that curve rather than more of the same.
This script measures where that curve actually puts each ion, so the decision about what to
spend is made on numbers rather than on extrapolation.

It also answers the real risk of the open-shell path: whether average-of-configuration
converges **reproducibly from a cold start** for the lanthanides, rather than only from a lucky
guess. Nothing here is seeded from a previous solution.

Every ion is bounded and every result is written as it is produced, so a run killed at its
budget still yields what it finished.

Run:  python tests/generate/amf_open_shell_cost.py [--contracted] [--only Ti,Ce]
      (with ``external/env.sh`` sourced). Writes ``temp/amf_open_shell_cost.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Sequence

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                                          # noqa: E402
from pyscf import gto                                                       # noqa: E402

import thermal                                                              # noqa: E402
from kuiva.amf.atomic import atomic_solution, clear_cache                   # noqa: E402
from kuiva.basis import registry                                            # noqa: E402
from kuiva.amf.backend import get_backend                                   # noqa: E402
from kuiva.amf.correction import validate_correction                        # noqa: E402
from kuiva.amf.decouple import amf_atomic_correction, x2c_decoupling        # noqa: E402
from kuiva.amf.pyscf_dhf import _density_anisotropy                         # noqa: E402

OUT = REPO / "temp/amf_open_shell_cost.json"
BASIS = "x2c-SVPall-2c"

#: Tried in order for elements ``BASIS`` does not cover — i.e. past Rn. the registry names the
#: Peterson ``cc-pVnZ-X2C`` sets as primary for actinides.
ACTINIDE_BASES = ("cc-pVDZ-X2C", "cc-pVTZ-X2C")

HARTREE_CM = 219474.6313632

#: The ions of the open-shell cost study, cheapest first so a budgeted run gets the most
#: information before it stops. ``None`` takes the default reference, which is the **trivalent
#: ion** for the f block and the neutral atom otherwise (:mod:`kuiva.amf.configuration`); the
#: f-block entries name it anyway, so the record says what was solved without a reader having
#: to know the policy.
#:
#: ⚠ The neutral Ce and Dy entries an earlier version carried are gone. They existed for the
#: sensitivity study, which Lu has since settled (13 ppm), and after the
#: f-block default changed ``("Ce", 0, None)`` would have silently duplicated the Ce(3+) entry.
IONS = (
    ("Ti", 3, "[Ar]3d1"),
    ("Ce", 3, "[Xe]4f1"),
    ("Yb", 3, "[Xe]4f13"),
    ("Dy", 3, "[Xe]4f9"),
    ("Bi", 0, None),
    ("U", 3, "[Rn]5f3"),
)


def basis_name_for(symbol: str) -> str:
    """The basis to use for one element, chosen by **coverage** rather than assumed.

    ⚠ The project default ``x2c-SVPall-2c`` is Karlsruhe and covers **H-Rn only**, so an
    actinide is not merely expensive in it — it does not exist, and PySCF raises a
    ``BasisNotFoundError`` from Basis Set Exchange. That is what killed an earlier run at U(3+)
    after five ions had already succeeded. The registry knows the coverage, so it is asked
    instead of guessed, and the actinide falls back to the Peterson ``cc-pVnZ-X2C`` set,
    the family named primary for actinides.
    """
    if registry.covers(BASIS, symbol):
        return BASIS
    for candidate in ACTINIDE_BASES:
        if registry.covers(candidate, symbol):
            return candidate
    raise ValueError(
        "no registered basis covers {}: {} stops at Rn and none of {} covers it either"
        .format(symbol, BASIS, ", ".join(ACTINIDE_BASES)))


def basis_for(symbol: str):
    name = basis_name_for(symbol)
    n = int(gto.charge(symbol))
    mol = gto.M(atom=[(symbol, (0.0, 0.0, 0.0))],
                basis=registry.resolve_for_pyscf(name, [symbol]), spin=n % 2, verbose=0)
    return mol._basis[symbol], int(mol.nao), name


def solver_mole(symbol: str, uncontract: bool):
    mol = gto.M(atom=[(symbol, (0.0, 0.0, 0.0))],
                basis=registry.resolve_for_pyscf(basis_name_for(symbol), [symbol]),
                spin=int(gto.charge(symbol)) % 2, verbose=0)
    return mol.decontract_basis(aggregate=True)[0] if uncontract else mol


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="", help="comma-separated element symbols")
    ap.add_argument("--contracted", action="store_true",
                    help="solve in the basis as given instead of decontracting; much cheaper, "
                         "at the cost of doing the X2C decoupling in a contracted basis "
                         "(decoupling in a contracted basis is advised against)")
    ap.add_argument("--max-wall", type=float, default=540.0,
                    help="total wall budget [s]; no new ion is started past it "
                         "(the ten-minute rule for ad-hoc runs)")
    args = ap.parse_args(argv)

    wanted = {s.strip().capitalize() for s in args.only.split(",") if s.strip()}
    uncontract = not args.contracted
    document: Dict = {"basis": BASIS, "uncontracted": uncontract,
                      "environment": thermal.describe_environment(), "records": {}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()

    for symbol, charge, configuration in IONS:
        if wanted and symbol not in wanted:
            continue
        elapsed = time.time() - started
        if elapsed > args.max_wall:
            print("[budget] {:.0f} s of {:.0f} s used; stopping before {}{:+d}".format(
                elapsed, args.max_wall, symbol, charge), flush=True)
            break
        key = "{}{:+d}/{}".format(symbol, charge, configuration or "default")
        clear_cache()
        try:
            # ⚠ Inside the try, deliberately. Resolving the basis used to sit above it, so an
            # element the basis does not cover aborted the **whole run** instead of being
            # recorded as one failed ion — U(3+) killed a run that had already finished five.
            # A failure must cost one record, not the remainder.
            basis, nao_contracted, basis_name = basis_for(symbol)
            with thermal.track_resources() as res:
                solution = atomic_solution(symbol, basis, configuration=configuration,
                                           uncontract=uncontract)
                # ⚠ Convergence alone is not the point. The open-shell cost study asks
                # for convergence robustness *and* forbids "a silently poor correction", and a
                # converged solve can still yield an unusable correction — that is exactly what
                # the singular-R bug did for the lanthanides. So the correction is built here
                # and its time-reversal residual recorded; `amf_atomic_correction` refuses
                # anything above `TIME_REVERSAL_LIMIT`, so a bad one shows up as a status.
                impl = get_backend("pyscf")
                amf = amf_atomic_correction(
                    solution, lambda dm: impl.coulomb_mean_field(solution, dm))
                validate_correction(amf.h_sf, amf.w,
                                    what="{} correction".format(symbol))
                x, _ = x2c_decoupling(solution)
            aniso = _density_anisotropy(solver_mole(symbol, uncontract),
                                        solution.density.ll)
            occ = np.asarray(solution.mo_occ)
            record = {
                "element": symbol, "charge": solution.charge, "basis": basis_name,
                "configuration": solution.configuration.canonical,
                "nao_contracted": nao_contracted, "nao_solver": solution.nao,
                "n4c": 4 * solution.nao, "e_tot": solution.e_tot,
                "converged": bool(solution.converged),
                "anisotropy": float(aniso),
                "fractional_occupations": sorted(
                    set(np.round(occ[(occ > 1e-12) & (occ < 1 - 1e-12)], 8).tolist())),
                "max_x": float(np.max(np.abs(x))),
                "correction": {
                    "max_dg": amf.scale,
                    "max_dh_sf": amf.spin_free_scale,
                    "max_dw": amf.spin_orbit_scale,
                    "tr_residual": amf.tr_residual,
                    "tr_residual_relative": amf.tr_residual_rel,
                    "cancellation": amf.cancellation,
                },
                "resources": res.as_dict(), "status": "ok",
            }
        except Exception as exc:                                            # noqa: BLE001
            record = {"element": symbol, "charge": charge,
                      "configuration": configuration,
                      "status": "{}: {}".format(type(exc).__name__, str(exc)[:200])}
        document["records"][key] = record
        OUT.write_text(json.dumps(document, indent=2, sort_keys=True))
        if record["status"] == "ok":
            c = record["correction"]
            print("{:16s} n4c {:4d}  E = {:15.6f}  conv={}  aniso={:.0e}  max|X|={:5.2f}  "
                  "max|dw|={:.3e}  TR-odd={:.1e} rel  cancel={:5.1f}  {}".format(
                      key, record["n4c"], record["e_tot"], record["converged"],
                      record["anisotropy"], record["max_x"], c["max_dw"],
                      c["tr_residual_relative"], c["cancellation"],
                      record["resources"].get("summary", "")), flush=True)
        else:
            print("{:16s} {}".format(key, record["status"]), flush=True)

    print("\nwrote {}".format(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
