"""Generate Tier-2 reference data with DIRAC (Tier 2).

DIRAC provides the *most rigorous* relativistic reference available here, and — crucially —
it can be run in a basis PySCF also has (``dyall.v2z`` <-> PySCF ``dyallv2z``, with the
primitive exponents already cross-checked in ``tests/reference/basis_crosscheck.json``). With
the basis held fixed, a Kuiva-vs-DIRAC discrepancy is attributable to the Hamiltonian and the
correlation treatment alone, which is the only way a cross-code number is worth anything.

Protocol
--------
* **Hamiltonian: X2C** (``.X2C``), not the full four-component Dirac-Coulomb. This is chosen
  deliberately: X2C is what Kuiva itself implements, so it is the closest possible
  comparison rather than the most expensive one. Four-component remains out of scope,
  but ``.DOSSSS`` can be requested here (``--hamiltonian dc``) to *quantify* what the 2c
  approximation costs — useful evidence, never a target.
* **Reference: average-of-configuration DHF** over the open shell. This is the natural
  analogue of a state-averaged reference: it makes no arbitrary choice among the degenerate
  open-shell configurations, so the multiplet structure comes entirely from the subsequent CI.
* **CI: KRCI** (LUCIAREL), a Kramers-restricted CI in exactly the open shell — the direct
  counterpart of Kuiva's CAS-CI in the spinor active space. All non-active occupied
  orbitals are ``.INACTIVE``.

What DIRAC contributes that OpenMolcas does not
-----------------------------------------------
Energies computed *without* the DKH2/AMFI approximations, and — because DIRAC finds the
atomic D(inf)h symmetry — an **Omega-resolved** spectrum: roots are requested per |Omega| =
|m_j| value. That is a far more structured reference than a flat list of energies, and it
maps directly onto the axial ``omega`` quantum-number labels the DMRG ``QuantumNumber`` tuple
is designed to carry later.

Setup traps worth recording (each cost a failed run while building this)
------------------------------------------------------------------------
* ``.CIROOTS`` in linear symmetry takes ``2*Omega`` with a ``g``/``u`` label, not a plain
  irrep number; a bare integer aborts with "Dinfh contains the inversion operation".
* ``.CI PROGRAM`` must have its value flush-left (``LUCIAREL``); a leading blank truncates it
  to ``LUCIARE`` and DIRAC rejects it as unknown.
* ``**MOLTRA .ACTIVE`` must cover the **inactive core as well as** the active shell. Excluding
  the core gives CI energies missing hundreds of Hartree, and then a segmentation fault.
* ``.CLOSED SHELL`` needs one count per fermion irrep (gerade, ungerade) once the molecule has
  an inversion centre.

References
----------
* DIRAC: T. Saue et al., J. Chem. Phys. 152, 204104 (2020), doi:10.1063/5.0004844;
  DIRAC26 (2026), http://www.diracprogram.org, doi:10.5281/zenodo.3572669.
* X2C in DIRAC (infinite-order two-component): M. Ilias, T. Saue, J. Chem. Phys. 126, 064102
  (2007), doi:10.1063/1.2436882.
* KRCI / LUCIAREL (Kramers-restricted GAS CI): S. Knecht, H. J. Aa. Jensen, T. Fleig,
  J. Chem. Phys. 132, 014108 (2010), doi:10.1063/1.3276157; T. Fleig, J. Olsen,
  L. Visscher, J. Chem. Phys. 119, 2963 (2003), doi:10.1063/1.1590636.
* Average-of-configuration DHF: J. Thyssen, Ph.D. thesis, Univ. of Southern Denmark (2001);
  T. Saue, H. J. Aa. Jensen, J. Chem. Phys. 111, 6211 (1999), doi:10.1063/1.479958.
* Dyall basis sets: K. G. Dyall, Theor. Chem. Acc. 135, 128 (2016) and companion papers;
  http://dirac.chem.sdu.dk.

Run:  python tests/generate/tier2_dirac.py [--only KEY,...]
(with ``external/env.sh`` sourced). Writes ``tests/reference/tier2_dirac.json``.
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

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kuiva.props import multiplet as mult_mod                 # noqa: E402
import systems as sysdef                                      # noqa: E402
import thermal                                                # noqa: E402

REF_OUT = REPO / "tests/reference/tier2_dirac.json"
DIRAC_ROOT = REPO / "external/install/dirac"
PAM = DIRAC_ROOT / "share/dirac/pam"

#: Uncontracted Dyall valence-double-zeta. Matches PySCF's ``dyallv2z`` exactly, which
#: is the whole point; the DZ level keeps a desktop run to seconds-to-minutes.
BASIS = "dyall.v2z"

DEGENERACY_TOL_CM = 1.0


@dataclass(frozen=True)
class DiracSpec:
    """DIRAC-specific setup for a system: shell structure and the roots to converge.

    Attributes
    ----------
    closed_g, closed_u : int
        Closed-shell **electron** counts per fermion irrep (gerade, ungerade).
    open_electrons, open_g, open_u : int
        Open-shell electrons and the number of open-shell **spinors** per irrep.
    inactive_g, inactive_u : int
        Inactive (frozen-occupied) **Kramers pairs** per irrep for KRCI.
    gas_orbitals : (int, int)
        Active Kramers pairs per irrep forming the single GAS space.
    ciroots : tuple of (str, int)
        ``("1u", 2)`` = two roots with 2*Omega = 1, ungerade. Omega labels, see the module
        docstring.
    moltra_emin, moltra_emax : float
        Energy window [Eh] selecting the orbitals transformed to the MO basis. Must span the
        inactive core *and* the active shell.
    """
    closed_g: int
    closed_u: int
    open_electrons: int
    open_g: int
    open_u: int
    inactive_g: int
    inactive_u: int
    gas_orbitals: Tuple[int, int]
    ciroots: Tuple[Tuple[str, int], ...]
    moltra_emin: float = -3000.0
    moltra_emax: float = -0.05
    amfi_charge: int = 0
    """Artificial charge for the AMFI mean-field summation only (``.AMFICH``).

    X2C in DIRAC takes its two-electron spin-orbit term from AMFI, whose *scalar atomic*
    SCF (RELSCF) is a single-determinant code that does not converge for some high-spin
    open-shell atoms - the manual names Yb explicitly, and Bi fails here too, aborting with
    "TOO MANY SCF ITERATIONS". A small positive charge fixes the atomic SCF while changing
    the mean-field contribution negligibly, since AMFI's large terms are core, not valence.
    Left at 0 wherever the atomic SCF converges unaided (Ce(3+)).
    """


#: Per-system DIRAC setups. The shell counts follow from the electron configuration: for a
#: lanthanide(3+) the [Xe] core splits 30 gerade (s, d) / 24 ungerade (p), and the 4f shell is
#: ungerade. Omega values enumerate the |m_j| levels of the resulting J multiplets.
DIRAC_SPECS: Dict[str, DiracSpec] = {
    # Ce(3+) 4f^1: 2F5/2 (Omega = 1/2, 3/2, 5/2) and 2F7/2 (+ Omega = 7/2) -> 7 Kramers pairs.
    "ce3p": DiracSpec(closed_g=30, closed_u=24, open_electrons=1, open_g=0, open_u=14,
                      inactive_g=15, inactive_u=12, gas_orbitals=(0, 7),
                      ciroots=(("1u", 2), ("3u", 2), ("5u", 2), ("7u", 1)),
                      moltra_emax=-0.90),
    # Yb(3+) 4f^13: the same 14-spinor space with one hole; ground multiplet 2F7/2.
    "yb3p": DiracSpec(closed_g=30, closed_u=24, open_electrons=13, open_g=0, open_u=14,
                      inactive_g=15, inactive_u=12, gas_orbitals=(0, 7),
                      ciroots=(("1u", 2), ("3u", 2), ("5u", 2), ("7u", 1)),
                      moltra_emax=-0.90, amfi_charge=4),
    # Bi 6p^3: [Xe]4f^14 5d^10 6s^2 = 80 closed electrons, 3 in the 6p shell.
    # The gerade/ungerade split must follow the *configuration*, not just add up to 80:
    # gerade = 6 s shells (12 e) + 3d,4d,5d (30 e) = 42; ungerade = 2p-5p (24 e) + 4f (14 e)
    # = 38. DIRAC does not check this - it simply fills the lowest spinors of each irrep, so
    # a wrong split (44/36) silently leaves the 4f shell one Kramers pair short and produces
    # a CI spectrum spanning millions of cm^-1 instead of the correct 20-state manifold.
    # 20 states = 4S3/2 + 2D3/2 + 2D5/2 + 2P1/2 + 2P3/2. Roots must be requested per Omega
    # according to how many J levels *contain* that Omega, i.e. 5 / 4 / 1 for Omega =
    # 1/2 / 3/2 / 5/2 (only 2D5/2 reaches Omega = 5/2). Asking for the wrong distribution
    # still yields 20 states and looks plausible, but silently drops the 2P3/2 Omega = 1/2
    # component and converges a spurious root at ~4e5 cm^-1 in its place.
    "bi": DiracSpec(closed_g=42, closed_u=38, open_electrons=3, open_g=0, open_u=6,
                    inactive_g=21, inactive_u=19, gas_orbitals=(0, 3),
                    ciroots=(("1u", 5), ("3u", 4), ("5u", 1)),
                    moltra_emax=-0.15, amfi_charge=2),
}


def dirac_available() -> bool:
    return PAM.is_file()


# --- input construction -----------------------------------------------------------------
def build_mol(system: sysdef.System) -> str:
    """A DIRAC ``.mol`` file (MOLECULE format) with one basis-set block per element."""
    from kuiva.basis import registry as reg
    by_element: Dict[str, List[Tuple[str, Tuple[float, float, float]]]] = {}
    for sym, xyz in system.atoms:
        by_element.setdefault(sym, []).append((sym, xyz))
    lines = ["INTGRL", f"{system.label}", f"{BASIS}",
             f"C{len(by_element):4d}              A"]
    for sym, atoms in by_element.items():
        lines.append(f"{float(reg.z_of(sym)):10.1f} {len(atoms):4d}")
        for i, (_, (x, y, z)) in enumerate(atoms, start=1):
            label = sym if len(atoms) == 1 else f"{sym}{i}"
            lines.append(f"{label:<6s}{x:16.10f}{y:16.10f}{z:16.10f}")
        lines.append(f"LARGE BASIS {BASIS}")
    lines.append("FINISH")
    return "\n".join(lines) + "\n"


def build_inp(system: sysdef.System, spec: DiracSpec, hamiltonian: str = "x2c") -> str:
    """A DIRAC input: AOC-DHF then KRCI in the open shell."""
    ham = ".X2C" if hamiltonian == "x2c" else ".DOSSSS"
    # RELSCF (AMFI's atomic SCF) defaults to 50 iterations and aborts the whole run if it
    # does not converge; raise it, and apply the documented artificial-charge workaround
    # where the atom needs it. Only relevant to the X2C path - 4c takes its spin-orbit
    # terms from the full two-electron operator and never calls AMFI.
    amfi = ""
    if hamiltonian == "x2c":
        amfi = "\n*AMFI\n.MXITER\n 200"
        if spec.amfi_charge:
            amfi += f"\n.AMFICH\n {spec.amfi_charge:+d}"
    roots = "\n".join(f".CIROOTS\n {label} {n}" for label, n in spec.ciroots)
    return f"""**DIRAC
.TITLE
 {system.label}: {hamiltonian.upper()} AOC-DHF + KRCI in the open shell
.WAVE FUNCTION
**HAMILTONIAN
{ham}{amfi}
**INTEGRALS
**WAVE FUNCTION
.SCF
.KR CI
*SCF
.CLOSED SHELL
 {spec.closed_g} {spec.closed_u}
.OPEN SHELL
 1
 {spec.open_electrons}/{spec.open_g},{spec.open_u}
.MAXITR
 80
*KRCI
.CI PROGRAM
LUCIAREL
.INACTIVE
 {spec.inactive_g} {spec.inactive_u}
.GAS SHELLS
 1
 {spec.open_electrons} {spec.open_electrons} / {spec.gas_orbitals[0]} {spec.gas_orbitals[1]}
{roots}
.MAX CI
 60
.MXCIVE
 60
**MOLTRA
.ACTIVE
 energy {spec.moltra_emin} {spec.moltra_emax} 0.001
*END OF
"""


# --- running and parsing -----------------------------------------------------------------
def run_dirac(system: sysdef.System, spec: DiracSpec, workdir: Path,
              hamiltonian: str = "x2c", timeout_s: int = 7200,
              memory_gb: float = 4.0) -> Tuple[Path, "thermal.RunResources", str]:
    """Run one system; returns (rundir, resource accounting, status).

    ``timeout_s`` is a wall-clock guard, deliberately generous: on a thermally clamped machine
    it would otherwise fire for cooling rather than for cost.
    """
    rundir = workdir / system.key
    if rundir.exists():
        shutil.rmtree(rundir)
    rundir.mkdir(parents=True)
    (rundir / f"{system.key}.mol").write_text(build_mol(system))
    (rundir / "krci.inp").write_text(build_inp(system, spec, hamiltonian))
    with thermal.track_resources() as res:
        status = _invoke(rundir, system, memory_gb, timeout_s)
    return rundir, res, status


def _invoke(rundir: Path, system: sysdef.System, memory_gb: float, timeout_s: int) -> str:
    try:
        # Size the static WORK array (LUCIAREL fails with "MEMGET ERROR, insufficient work
        # space" at the default). Deliberately *not* --ag: capping the dynamically
        # allocatable pool makes KRCI's set_hop_dbg abort with a fatal allocation error at
        # whatever the cap is, whereas leaving it uncapped succeeds.
        proc = subprocess.run([str(PAM), "--noarch", f"--gb={memory_gb}", "--inp=krci.inp",
                               f"--mol={system.key}.mol"],
                              cwd=rundir, env=dict(os.environ), capture_output=True,
                              text=True, timeout=timeout_s)
        (rundir / "pam.log").write_text(proc.stdout + "\n" + proc.stderr)
        return "ok" if proc.returncode == 0 else f"pam rc={proc.returncode}"
    except subprocess.TimeoutExpired:
        return f"timeout after {timeout_s} s wall"


#: Lines worth keeping when a run fails, so the committed reference records *why* a system
#: has no DIRAC data rather than silently omitting it.
_ERROR_PATTERNS = ("Fatal memory allocation error", "MEMGET ERROR", "SELF CONSISTENCE",
                   "TOO MANY SCF ITERATIONS", "input error", "does not match",
                   "SIGSEGV", "FATAL ERROR")


def dirac_error(out_path: Path) -> str:
    """The most informative error line from a failed DIRAC run (for the record)."""
    try:
        text = out_path.read_text(errors="replace")
    except OSError:
        return "output unreadable"
    hits = [ln.strip() for ln in text.splitlines()
            if any(p in ln for p in _ERROR_PATTERNS)]
    return hits[-1] if hits else "no recognised error message"


_ROOT_HEADER = re.compile(r"eigenstate\(s\) for MJ-value\s*:\s*(\d+)([gu])?/2")
_CI_ENERGIES = re.compile(r"Final CI energies\s*=\s*(.*)")


def parse_output(out_path: Path) -> Dict:
    """Extract the Omega-resolved KRCI spectrum from a DIRAC output.

    DIRAC prints one ``Final CI energies`` line per Omega block, in the order the blocks were
    requested. Each energy is a **Kramers pair**, i.e. two degenerate states, so the state
    list is built by counting every root twice — which is exactly the Kramers degeneracy
    Kuiva's general-complex CI must reproduce numerically.
    """
    text = out_path.read_text(errors="replace")
    omegas = [f"{m.group(1)}{m.group(2) or ''}" for m in _ROOT_HEADER.finditer(text)]
    # DIRAC prints at most four energies per line and wraps the rest onto continuation
    # lines. Reading only the first line silently truncates any block with five or more
    # roots, which shows up as a plausible-looking spectrum that is simply missing states.
    lines = text.splitlines()
    blocks: List[List[float]] = []
    for i, line in enumerate(lines):
        m = _CI_ENERGIES.match(line.strip()) or _CI_ENERGIES.search(line)
        if not m:
            continue
        vals = [float(v) for v in m.group(1).split()]
        for cont in lines[i + 1:]:
            toks = cont.split()
            if not toks:
                break
            try:
                vals.extend(float(t) for t in toks)
            except ValueError:
                break
        blocks.append(vals)
    blocks = [b for b in blocks if b]
    if not blocks:
        raise ValueError("no 'Final CI energies' found in the DIRAC output")
    # One block per requested Omega value, in request order. They must NOT be de-duplicated:
    # different Omega blocks legitimately carry identical energies (the Omega = 1/2, 3/2, 5/2
    # levels of a single 2F5/2 multiplet are degenerate), and collapsing them would silently
    # destroy the very degeneracy structure this reference exists to check.
    if len(blocks) != len(omegas):
        raise ValueError(f"{len(blocks)} CI energy blocks but {len(omegas)} Omega headers")

    scf = re.findall(r"Total energy\s*:\s*(-?\d+\.\d+)", text)
    per_omega: List[Dict] = []
    energies: List[float] = []
    for omega, vals in zip(omegas, blocks):
        per_omega.append({"omega_2x": omega, "n_roots": len(vals),
                          "energies": [float(v) for v in vals]})
        energies.extend(list(vals) * 2)              # each root is a Kramers pair
    e = np.array(sorted(energies))
    mults = mult_mod.analyse_spectrum(e, tol_cm=DEGENERACY_TOL_CM)
    return {
        "e_scf_total": float(scf[-1]) if scf else None,
        "n_soc_states": int(e.size),
        "e_soc_total_lowest": float(e.min()),
        "soc_rel_cm": [round(float(x), 4) for x in (e - e.min()) * mult_mod.HARTREE_TO_CM],
        "degeneracy_pattern": list(mult_mod.degeneracy_pattern(mults)),
        "multiplets": [{"size": m.size, "energy_cm": round(m.energy_cm, 4),
                        "spread_cm": round(m.spread_cm, 6)} for m in mults],
        "omega_blocks": per_omega,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="", help="comma-separated system keys")
    ap.add_argument("--workdir", default="")
    ap.add_argument("--hamiltonian", default="x2c", choices=["x2c", "dc"],
                    help="x2c (default, matches Kuiva) or dc (4c Dirac-Coulomb, for scale)")
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--memory-gb", type=float, default=4.0, help="DIRAC WORK array size")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args(argv)

    if not dirac_available():
        print(f"DIRAC not found at {PAM}", file=sys.stderr)
        return 2
    workdir = Path(args.workdir) if args.workdir else REPO / "temp" / "tier2_dirac"
    workdir.mkdir(parents=True, exist_ok=True)

    out: Dict = {"schema": 2, "generator": "tests/generate/tier2_dirac.py", "code": "DIRAC",
                 "basis": BASIS,
                 "hamiltonian": f"{args.hamiltonian.upper()} AOC-DHF + KRCI (LUCIAREL)",
                 "degeneracy_tol_cm": DEGENERACY_TOL_CM,
                 "environment": thermal.describe_environment(), "records": {}}
    if args.merge and REF_OUT.is_file():
        out = json.loads(REF_OUT.read_text())
        out.setdefault("records", {})

    keys = [k for k in args.only.split(",") if k] or None
    rc = 0
    for system in sysdef.for_tier2("dirac"):
        if keys and system.key not in keys:
            continue
        spec = DIRAC_SPECS.get(system.key)
        if spec is None:
            print(f"[dirac]  {system.key:12s} no DiracSpec defined - skipped", flush=True)
            continue
        rundir, res, status = run_dirac(system, spec, workdir,
                                        hamiltonian=args.hamiltonian, timeout_s=args.timeout,
                                        memory_gb=args.memory_gb)
        rec: Dict = {"key": system.key, "label": system.label, "basis": BASIS,
                     "hamiltonian": args.hamiltonian, "resources": res.as_dict(),
                     "status": status}
        out_file = rundir / f"krci_{system.key}.out"
        if out_file.is_file():
            try:
                rec.update(parse_output(out_file))
                rec["status"] = "ok"
                if rec["n_soc_states"] != system.soc_states:
                    rec["status"] = (f"state-count mismatch: got {rec['n_soc_states']}, "
                                     f"expected {system.soc_states}")
            except Exception as exc:                            # noqa: BLE001
                rec["status"] = f"parse failed: {type(exc).__name__}: {exc}"
                rec["diagnostic"] = dirac_error(out_file)
        else:
            rec["status"] = f"no output file ({status})"
        out["records"][system.key] = rec
        if rec["status"] != "ok":
            rc = 1
        print(f"[dirac]  {system.key:12s} {rec['status']:28s} "
              f"states={rec.get('n_soc_states', '-')} "
              f"pattern={rec.get('degeneracy_pattern', '-')} ({res.summary()})", flush=True)
        if res.throttled:
            # WARNING: completed, but the wall time above is cooling, not cost.
            print(f"  [warn] {system.key}: CPU thermally clamped for "
                  f"{100 * (res.throttle_fraction or 0):.0f}% of the run", flush=True)

    REF_OUT.parent.mkdir(parents=True, exist_ok=True)
    REF_OUT.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"\nwrote {REF_OUT.relative_to(REPO)}  ({len(out['records'])} records)")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
