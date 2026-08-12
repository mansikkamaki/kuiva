"""The method-decomposition table: one-electron X2C, X2CAMF, DKH2+AMFI, four-component.

This is the table that answers *"why does our number differ from the published OpenMolcas
one"* with a measurement instead of a claim — for the p-block
series where every column can be produced for the **same observable in the same basis**.

⚠ This is NOT a controlled single-variable comparison, and saying so is half the deliverable
--------------------------------------------------------------------------------------------
OpenMolcas differs from Kuiva in **three** ways at once and the table cannot separate them:

* **Hamiltonian** — DKH2 against X2C. Different decoupling, different order.
* **Two-electron spin-orbit treatment** — Breit-Pauli AMFI (Hess, Marian, Wahlgren, Gropen
  1996) against X2CAMF (Liu & Cheng 2018). ⚠ AMFI carries **no two-electron scalar picture
  change at all**, which is the same structural gap the plugin's spin-dependent-only variant
  has (``tests/test_x2camf_plugin.py``) and which is recorded as a
  distinguishing feature of X2CAMF.
* **How the splitting is obtained** — a RASSCF/RASSI-SO **term** energy against a
  one-particle **spinor** splitting from a self-consistent two-component SCF.

The third is why the atom set is what it is. Every system here is a **p⁵ cation**: a single
hole outside a closed shell, which is the one open-shell case where the measured ²P₃/₂-²P₁/₂
term splitting *is* a one-particle quantity (the rule that denies Bi an
experimental anchor). A ``(5e, 3o)`` active space over the valence ``p`` shell spans exactly
the three components of ²P and contains **no correlation at all**, so the RASSI-SO splitting
is, to first order, the same ``(3/2) zeta`` the spinor splitting is. That reduces the third
confound to the orbitals it is evaluated over, and leaves the first two — which are the point.

⚠ **What the table then measures, and it is a large number.** DKH2+AMFI comes out **17-19%
below** X2C+X2CAMF for the same ion in the same basis, while X2CAMF reproduces
**four-component Dirac-Coulomb in that same basis to 0.4% (Ne) and 1.3% (Ar)**. Since a
four-component calculation performs no picture change, that second figure is the statement
that the X2CAMF column is right about what it claims to be. ⚠ **It is also further from
experiment**, and that is not a contradiction: a mean-field one-particle splitting is not a
correlated observable, and Breit-Pauli AMFI errs in the direction correlation would move the
result. Do not read the experimental column as ranking the two methods — read it as the size
of what is still missing from *both*, which is electron correlation, and which becomes
checkable when there is a CI to compute atomic multiplet energies with .

What each column is
-------------------
========================= ===============================================================
``one_electron_x2c``      Kuiva, X2C one-electron only — the error X2CAMF exists to remove
``kuiva_coulomb/gaunt``   Kuiva X2CAMF, four-component Dirac-Coulomb(-Gaunt) atomic mean
                          field, self-consistent two-component AOC SCF
``molcas_dkh2_amfi``      OpenMolcas: DKH2 + AMFI, RASSCF(p^5)/RASSI-SO term splitting
``four_component_*``      Kuiva's own four-component solve for the same ion, contracted
                          **and** primitive
``experiment``            NIST ASD, the ²P₃/₂-²P₁/₂ interval of the singly charged ion
========================= ===============================================================

⚠ **Both four-component columns are recorded and the contracted one is the comparison.**
Quoting a contracted two-component number against a primitive four-component reference is the
trap documented at the mean field — it reads as up to 16% of method error that is
really basis-set truncation. OpenMolcas works in the contracted ANO-RCC basis, so that is the
space the comparison lives in; the primitive column is kept because it is the converged answer
and the gap between the two *is* the basis error, stated rather than hidden.

⚠ **The atomic mean field is taken over the neutral atom on both sides.** Kuiva's default
reference for a p-block element is the neutral atom  and AMFI's atomic
mean field is likewise not the ion's. The measured sensitivity to that choice is 0.21% for a
3d ion and 13 ppm for 4f, so it is not what any difference in this table is.

Basis: **ANO-RCC-VDZP** everywhere — OpenMolcas's native, generally contracted set, bundled
with PySCF as ``ano-rcc-vdzp`` and already primitive-checked against the OpenMolcas library in
``tests/reference/basis_crosscheck.json`` .

Cost: seconds per atom for Ne and Ar on both sides; Kr and Xe are minutes on the **Kuiva**
side (the uncontracted four-component atomic solve grows as ``n4c^4``) and are not in the
default set .

Run:  python tests/generate/x2camf_molcas_amfi.py [--only Ne,Ar] (with external/env.sh
      sourced). Writes ``tests/reference/x2camf_molcas_amfi.json``.

References
----------
* AMFI: B. A. Hess, C. M. Marian, U. Wahlgren, O. Gropen, Chem. Phys. Lett. 251, 365 (1996),
  doi:10.1016/0009-2614(96)00119-4.
* RASSI spin-orbit state interaction: P. A. Malmqvist, B. O. Roos, B. Schimmelpfennig,
  Chem. Phys. Lett. 357, 230 (2002), doi:10.1016/S0009-2614(02)00498-0.
* Douglas-Kroll-Hess: M. Douglas, N. M. Kroll, Ann. Phys. 82, 89 (1974); B. A. Hess,
  Phys. Rev. A 33, 3742 (1986), doi:10.1103/PhysRevA.33.3742.
* OpenMolcas: G. Li Manni et al., J. Chem. Theory Comput. 19, 6933 (2023),
  doi:10.1021/acs.jctc.3c00182.
* X2CAMF: J. Liu, L. Cheng, J. Chem. Phys. 148, 144108 (2018), doi:10.1063/1.5023750.
* ANO-RCC basis sets: B. O. Roos, R. Lindh, P.-A. Malmqvist, V. Veryazov, P.-O. Widmark,
  J. Phys. Chem. A 108, 2851 (2004), doi:10.1021/jp031064+.
* Experimental fine structure: A. Kramida, Yu. Ralchenko, J. Reader and NIST ASD Team, NIST
  Atomic Spectra Database (version 5.11), https://physics.nist.gov/asd.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                                          # noqa: E402
from pyscf import gto                                                       # noqa: E402

import thermal                                                              # noqa: E402
from amf_sensitivity import spectrum_of                                     # noqa: E402
from kuiva.amf import amf_correction                                        # noqa: E402
from kuiva.amf.atomic import atomic_solution, clear_cache                   # noqa: E402
from kuiva.amf.configuration import AtomicConfiguration                     # noqa: E402

OUT = REPO / "tests/reference/x2camf_molcas_amfi.json"
MOLCAS_ROOT = REPO / "external/install/openmolcas"
HARTREE_CM = 219474.6313632

#: OpenMolcas's own name and PySCF's for the same set. Content-matched in
#: ``tests/reference/basis_crosscheck.json``.
MOLCAS_BASIS = "ANO-RCC-VDZP"
PYSCF_BASIS = "ano-rcc-vdzp"


@dataclass(frozen=True)
class Ion:
    """A singly charged p-block cation: one hole outside a closed shell.

    ``inactive`` is the number of doubly occupied orbitals below the valence ``p`` shell, set
    **explicitly** in the RASSCF input rather than left to OpenMolcas, for the reason
    ``tier2_molcas.py`` gives: it fixes the electron count and hence the charge unambiguously,
    which is what makes the reference reproducible.

    ``experiment_cm`` is the ²P₃/₂-²P₁/₂ interval of the ion from NIST ASD. ⚠ It is quoted
    here — unlike for Bi  — precisely because a single hole in a closed
    shell has no term structure beyond the spin-orbit splitting itself, so the measured
    interval and the computed one-particle splitting are the same observable.
    """

    symbol: str
    shell: str
    inactive: int
    experiment_cm: float


IONS: Tuple[Ion, ...] = (
    Ion("Ne", "2p", 2, 780.4),        # Ne II  1s2 2s2 2p5
    Ion("Ar", "3p", 6, 1431.6),       # Ar II  [Ne] 3s2 3p5
    Ion("Kr", "4p", 15, 5370.1),      # Kr II  [Ar] 3d10 4s2 4p5
    Ion("Xe", "5p", 24, 10537.0),     # Xe II  [Kr] 4d10 5s2 5p5
)
BY_SYMBOL = {i.symbol: i for i in IONS}
DEFAULT_IONS = ("Ne", "Ar")


def molcas_available() -> bool:
    return (MOLCAS_ROOT / "pymolcas").is_file()


# --- OpenMolcas ---------------------------------------------------------------------------
def build_input(ion: Ion) -> str:
    """DKH2 + AMFI, RASSCF over the three ²P components, then RASSI-SO.

    Nothing here asks for DKH2 or AMFI by keyword: OpenMolcas switches both on for an ANO-RCC
    basis, and the generated output is checked for the two lines that say so
    (:func:`parse_output`) rather than the behaviour being assumed.
    """
    return ("&GATEWAY\n"
            " Coord\n 1\n {} cation, DKH2+AMFI reference\n"
            " {} 0.0000000000 0.0000000000 0.0000000000\n"
            " Basis = {}\n"
            " Group = C1\n"
            " AngMom = 0.0 0.0 0.0\n"
            " RICD\n"
            "&SEWARD\n"
            "&RASSCF\n"
            " Title = {}+ 2P\n"
            " Spin = 2\n"
            " Charge = 1\n"
            " Inactive = {}\n"
            " Ras2 = 3\n"
            " nActEl = 5 0 0\n"
            " CiRoot = 3 3 1\n"
            "&RASSI\n"
            " Nr of JobIphs = 1 3\n"
            " 1 2 3\n"
            " SpinOrbit\n"
            " EJob\n").format(ion.symbol, ion.symbol, MOLCAS_BASIS, ion.symbol,
                              ion.inactive)


_SO_ROW = re.compile(r"^\s+(\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s")


def parse_output(path: Path) -> Dict:
    """Read the six spin-orbit eigenvalues and the two Hamiltonian statements.

    ⚠ The degeneracy pattern is checked, not just the splitting. A ²P ion **must** come out
    4 + 2 (J = 3/2 below J = 1/2 for a more-than-half-filled shell); anything else means the
    active space or the charge was not what the input said, and the splitting would then be a
    plausible number for the wrong state.
    """
    text = path.read_text(errors="replace")
    if "Happy landing" not in text:
        raise ValueError("OpenMolcas did not finish ({})".format(path.name))
    if "Relativistic Douglas-Kroll-Hess integrals" not in text:
        raise ValueError("the run was not relativistic — no DKH integrals were computed")
    if "Atomic mean-field integrals" not in text:
        raise ValueError("no AMFI integrals were computed; the spin-orbit operator is absent")

    block = text.split("Eigenvalues of complex Hamiltonian:")[-1]
    energies: List[float] = []
    for line in block.splitlines():
        m = _SO_ROW.match(line)
        if m and int(m.group(1)) == len(energies) + 1:
            energies.append(float(m.group(4)))       # cm^-1 relative to the lowest level
        elif energies and not line.strip():
            if len(energies) >= 6:
                break
    if len(energies) < 6:
        raise ValueError("found {} spin-orbit states, expected 6".format(len(energies)))
    energies = energies[:6]
    lower = [e for e in energies if abs(e - energies[0]) < 1.0]
    upper = [e for e in energies if abs(e - energies[0]) >= 1.0]
    if len(lower) != 4 or len(upper) != 2:
        raise ValueError("the spin-orbit manifold split {} + {}, not 4 + 2 — this is not a "
                         "²P ion".format(len(lower), len(upper)))
    return {"so_energies_cm": energies,
            "degeneracies": [len(lower), len(upper)],
            "splitting_cm": float(np.mean(upper) - np.mean(lower)),
            "hamiltonian": "DKH2 (order 2, EXP parametrization) + AMFI spin-orbit"}


def run_molcas(ion: Ion, workdir: Path, timeout_s: int, memory_mb: int) -> Tuple[Path, str]:
    rundir = workdir / ion.symbol
    if rundir.exists():
        shutil.rmtree(rundir)
    (rundir / "work").mkdir(parents=True)
    name = "{}p".format(ion.symbol.lower())
    inp = rundir / "{}.input".format(name)
    inp.write_text(build_input(ion))
    env = dict(os.environ)
    env.update({"MOLCAS": str(MOLCAS_ROOT), "MOLCAS_NPROCS": "1",
                "MOLCAS_MEM": str(memory_mb), "MOLCAS_WORKDIR": str(rundir / "work"),
                "MOLCAS_PRINT": "2"})
    out = rundir / "{}.out".format(name)
    try:
        proc = subprocess.run([str(MOLCAS_ROOT / "pymolcas"), inp.name], cwd=rundir, env=env,
                              capture_output=True, text=True, timeout=timeout_s)
        out.write_text(proc.stdout + "\n" + proc.stderr)
        return out, "ok" if proc.returncode == 0 else "pymolcas rc={}".format(proc.returncode)
    except subprocess.TimeoutExpired:
        return out, "timeout after {} s wall".format(timeout_s)


# --- Kuiva ---------------------------------------------------------------------------------
def kuiva_columns(ion: Ion) -> Dict:
    """Every column that does not need an external program, for the same ion and basis.

    The ion's own configuration (``p^5``) is what the two-component SCF is solved in; the
    **reference configuration of the mean field** is the neutral atom, which is Kuiva's
    default for a p-block element and the closest counterpart to what AMFI does.
    """
    clear_cache()
    symbol = ion.symbol
    state = AtomicConfiguration.for_oxidation_state(symbol, 1)
    mol = gto.M(atom=[(symbol, (0.0, 0.0, 0.0))], basis=PYSCF_BASIS, charge=1,
                spin=(int(gto.charge(symbol)) - 1) % 2, verbose=0)

    columns: Dict[str, float] = {
        "one_electron_x2c": _splitting(spectrum_of(mol, state, None)),
    }
    for interaction in ("coulomb", "gaunt"):
        correction = amf_correction(mol, method="x2camf", interaction=interaction)
        columns["kuiva_" + interaction] = _splitting(spectrum_of(mol, state, correction))
    # ⚠ Both, and the contracted one is the like-for-like comparison.
    for label, uncontract in (("contracted", False), ("primitive", True)):
        solution = atomic_solution(symbol, PYSCF_BASIS, configuration=state,
                                   interaction="coulomb", uncontract=uncontract)
        columns["four_component_" + label] = float(solution.shell_splitting(6) * HARTREE_CM)
    return {"splitting_cm": columns,
            "state": state.canonical,
            "mean_field_reference": AtomicConfiguration.ground(symbol).canonical,
            "construction": ("self-consistent two-component average-of-configuration SCF in "
                             "the contracted molecular basis; the mean field is decoupled in "
                             "the primitive basis and contracted back")}


def _splitting(energies: np.ndarray, width: int = 6) -> float:
    valence = np.sort(np.asarray(energies))[-width:]
    return float((valence[-1] - valence[0]) * HARTREE_CM)


# --- driver ---------------------------------------------------------------------------------
def _new_document() -> Dict:
    return {
        "schema": 1,
        "generator": "tests/generate/x2camf_molcas_amfi.py",
        "code": "OpenMolcas v26.06 (DKH2 + AMFI, RASSCF/RASSI-SO) and Kuiva",
        "purpose": "method-decomposition table for the two-electron spin-orbit treatment",
        "basis": {"molcas": MOLCAS_BASIS, "pyscf": PYSCF_BASIS},
        "observable": ("the ²P₃/₂-²P₁/₂ splitting of a singly charged p-block cation: one "
                       "hole outside a closed shell, and therefore the one open-shell case "
                       "where a term splitting and a one-particle splitting are the same "
                       "observable "),
        "confound": ("NOT a controlled single-variable comparison. OpenMolcas differs from "
                     "Kuiva in the Hamiltonian (DKH2 vs X2C), in the two-electron spin-orbit "
                     "treatment (Breit-Pauli AMFI, which has no two-electron scalar picture "
                     "change at all, vs X2CAMF) and in how the splitting is obtained "
                     "(RASSI-SO term energy vs one-particle spinor splitting). The third is "
                     "reduced to the choice of orbitals by taking a p^5 cation, whose (5e,3o) "
                     "active space spans exactly the three components of 2P and holds no "
                     "correlation; the first two remain and this table cannot separate them."),
        "environment": thermal.describe_environment(),
        "records": {},
    }


def _write(document: Dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(document, indent=2, sort_keys=True))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default=",".join(DEFAULT_IONS))
    ap.add_argument("--workdir", default="")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--memory-mb", type=int, default=3000)
    ap.add_argument("--max-wall", type=float, default=540.0)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args(argv)

    if not molcas_available():
        print("OpenMolcas not found at {}".format(MOLCAS_ROOT), file=sys.stderr)
        return 2
    workdir = Path(args.workdir) if args.workdir else REPO / "temp" / "x2camf_molcas_amfi"
    workdir.mkdir(parents=True, exist_ok=True)

    document = _new_document()
    if not args.fresh and OUT.is_file():
        try:
            document["records"] = json.loads(OUT.read_text()).get("records", {})
        except ValueError:
            pass

    started = time.time()
    rc = 0
    for symbol in [s.strip().capitalize() for s in args.only.split(",") if s.strip()]:
        ion = BY_SYMBOL.get(symbol)
        if ion is None:
            print("unknown ion {!r}".format(symbol), file=sys.stderr)
            rc = 1
            continue
        if time.time() - started > args.max_wall:
            print("[budget] stopping before {}".format(symbol), flush=True)
            break
        record: Dict = {"element": symbol, "charge": 1, "valence_shell": ion.shell,
                        "experiment_cm": ion.experiment_cm}
        with thermal.track_resources() as res:
            out_file, status = run_molcas(ion, workdir, args.timeout, args.memory_mb)
            try:
                molcas = parse_output(out_file)
                record["molcas"] = molcas
                record.update(kuiva_columns(ion))
                record["splitting_cm"]["molcas_dkh2_amfi"] = molcas["splitting_cm"]
                record["splitting_cm"]["experiment"] = ion.experiment_cm
                record["status"] = "ok"
            except Exception as exc:                                        # noqa: BLE001
                record["status"] = "{}: {}".format(type(exc).__name__, exc)
                record["molcas_status"] = status
                rc = 1
        record["resources"] = res.as_dict()
        document["records"][symbol] = record
        _write(document)                              # incremental, within the ten-minute ad-hoc budget
        if record["status"] == "ok":
            s = record["splitting_cm"]
            print("[amfi] {:3s}  1e {:9.2f} | X2CAMF {:9.2f} (+G {:9.2f}) | DKH2+AMFI "
                  "{:9.2f} | 4c {:9.2f} | exp {:9.2f} cm^-1  ({})".format(
                      symbol, s["one_electron_x2c"], s["kuiva_coulomb"], s["kuiva_gaunt"],
                      s["molcas_dkh2_amfi"], s["four_component_contracted"],
                      s["experiment"], res.summary()), flush=True)
        else:
            print("[amfi] {:3s}  {}".format(symbol, record["status"]), flush=True)

    _write(document)
    print("\nwrote {}  ({} records)".format(OUT.relative_to(REPO), len(document["records"])))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
