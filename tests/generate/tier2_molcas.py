"""Generate Tier-2 reference data with OpenMolcas (Tier 2).

OpenMolcas is used for the one thing it does that PySCF cannot: **RASSI-SO**, which couples
spin-free RASSCF states with the AMFI spin-orbit operator and yields, in the basis of the SOC
eigenstates, exactly the objects Kuiva dumps — the effective Hamiltonian
(the SOC state energies) and the magnetic-moment operator components. Those are read from the
machine-readable ``$Project.aniso`` interface file (``$eso``, ``$magn_*``) rather than scraped
from the printed output, so the reference carries full precision and no parsing ambiguity.
That file is OpenMolcas's own contract with its property code, i.e. the exact counterpart of
what ``props/dump.py`` will write for Kuiva.

(The ``rassi.h5`` datasets ``SOS_ANGMOM_*``/``SOS_SPIN_*`` look like the obvious source but
are written as **all zeros** unless extra RASSI keywords are supplied — taking the data from
there silently yields a reference with vanishing magnetic moments.)

This runs **once**; the parsed results are committed to ``tests/reference/`` and the
fast suite compares against the committed file without OpenMolcas being installed.

What is and is not comparable
-----------------------------
Kuiva and OpenMolcas will **not** agree to many digits, and the reference is built so that
this is not mistaken for an error. The methods genuinely differ:

* **Hamiltonian.** OpenMolcas uses DKH2 with the atomic mean-field (AMFI) spin-orbit operator;
  Kuiva uses X2C with SOC entering at the CI step. Expect SOC splittings to differ by a
  few percent.
* **Orbitals.** OpenMolcas optimises *separate* orbitals per spin multiplicity and lets RASSI
  couple the resulting non-orthogonal state sets; Kuiva state-averages one spinor orbital set
  over the whole manifold. For the two-multiplicity systems (``bi``, ``tlh``) this
  is a real methodological difference, not a tolerance issue.
* **Phases.** Arbitrary on both sides, and degenerate states mix arbitrarily.

So the stored reference deliberately leads with the quantities that *are* well defined:
degeneracy patterns (fixed by symmetry — these must match exactly), relative energies in
cm^-1, and the phase-invariant moment tensors and g values of
:mod:`kuiva.props.multiplet`. Absolute energies are stored too, but only as provenance.

References
----------
* OpenMolcas: G. Li Manni et al., J. Chem. Theory Comput. 19, 6933 (2023),
  doi:10.1021/acs.jctc.3c00182; F. Aquilante et al., J. Chem. Phys. 152, 214117 (2020).
* RASSI / spin-orbit state interaction: P. A. Malmqvist, B. O. Roos, B. Schimmelpfennig,
  Chem. Phys. Lett. 357, 230 (2002), doi:10.1016/S0009-2614(02)00498-0.
* AMFI atomic mean-field spin-orbit integrals: B. A. Hess, C. M. Marian, U. Wahlgren,
  O. Gropen, Chem. Phys. Lett. 251, 365 (1996), doi:10.1016/0009-2614(96)00119-4.
* Douglas-Kroll-Hess: M. Douglas, N. M. Kroll, Ann. Phys. 82, 89 (1974); B. A. Hess,
  Phys. Rev. A 33, 3742 (1986), doi:10.1103/PhysRevA.33.3742.
* Cholesky decomposition of the ERIs (``RICD``): F. Aquilante, P.-A. Malmqvist,
  T. B. Pedersen, A. Ghosh, B. O. Roos, J. Chem. Theory Comput. 4, 694 (2008),
  doi:10.1021/ct700263h.
* SINGLE_ANISO pseudospin/g-tensor analysis: L. F. Chibotaru, L. Ungur, J. Chem. Phys. 137,
  064112 (2012), doi:10.1063/1.4739763.

Run:  python tests/generate/tier2_molcas.py [--only KEY,...] [--workdir DIR]
(with ``external/env.sh`` sourced). Writes ``tests/reference/tier2_molcas.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kuiva.props import multiplet as mult_mod                 # noqa: E402
import systems as sysdef                                      # noqa: E402
import thermal                                                # noqa: E402

REF_OUT = REPO / "tests/reference/tier2_molcas.json"
MOLCAS_ROOT = REPO / "external/install/openmolcas"

#: ANO-RCC contraction level. VDZP keeps every system on this list to a few hundred basis
#: functions, which is what makes a desktop Tier-2 run possible at all. The test is
#: correctness of structure, not basis-set accuracy.
BASIS = "ANO-RCC-VDZP"

#: Degeneracy tolerance for grouping SOC states [cm^-1]. Well above converged-CI noise and
#: well below any physical splitting in these systems.
DEGENERACY_TOL_CM = 1.0


def molcas_available() -> bool:
    return (MOLCAS_ROOT / "pymolcas").is_file()


# --- input construction -----------------------------------------------------------------
def rasscf_block(system: sysdef.System, mult: int, nroots: int, jobiph: int) -> str:
    """A ``&RASSCF`` block for one spin multiplicity, writing a numbered JOBIPH for RASSI.

    ``Inactive`` is set explicitly rather than left to OpenMolcas: it fixes the electron
    count (and hence the charge) unambiguously, which is what makes the reference
    reproducible.
    """
    inactive = (system.nelectron - system.nelecas) // 2
    return (f"&RASSCF\n"
            f" Title = {system.key} mult{mult}\n"
            f" Spin = {mult}\n"
            f" Inactive = {inactive}\n"
            f" Ras2 = {system.ncas}\n"
            f" nActEl = {system.nelecas} 0 0\n"
            f" CiRoot = {nroots} {nroots} 1\n"
            f" OutOrbitals\n"
            f" Natural = {nroots}\n"
            f">> COPY $Project.JobIph JOB{jobiph:03d}\n")


def build_input(system: sysdef.System) -> str:
    """The full OpenMolcas input: SEWARD, one RASSCF per multiplicity, then RASSI-SO."""
    lines = [f"&GATEWAY", " Coord", f" {len(system.atoms)}", f" {system.label}"]
    for sym, (x, y, z) in system.atoms:
        lines.append(f" {sym} {x:.10f} {y:.10f} {z:.10f}")
    lines += [f" Basis = {BASIS}",
              " Group = C1",
              # Origin for the angular-momentum integrals that become the magnetic moment.
              " AngMom = 0.0 0.0 0.0",
              # Cholesky ERIs: the general fallback and what keeps this affordable.
              " RICD",
              "&SEWARD"]
    mults = sorted(system.nroots)
    for i, m in enumerate(mults, start=1):
        lines.append(rasscf_block(system, m, system.nroots[m], i))
    nroots_list = " ".join(str(system.nroots[m]) for m in mults)
    lines.append("&RASSI")
    lines.append(f" Nr of JobIphs = {len(mults)} {nroots_list}")
    for m in mults:
        lines.append(" " + " ".join(str(k + 1) for k in range(system.nroots[m])))
    lines += [" SpinOrbit", " EJob"]
    # SINGLE_ANISO is what writes $Project.aniso, the machine-readable file this script
    # parses. Its own pseudospin analysis is not used (and may legitimately decline to run
    # for a non-Kramers system) - only the interface file it emits matters here.
    lines += ["&SINGLE_ANISO"]
    return "\n".join(lines) + "\n"


# --- running ------------------------------------------------------------------------------
def run_molcas(system: sysdef.System, workdir: Path, nprocs: int = 1,
               memory_mb: int = 3000, timeout_s: int = 7200
               ) -> Tuple[Path, "thermal.RunResources", str]:
    """Run one system; returns (rundir, resource accounting, status).

    ``timeout_s`` is a **wall-clock** guard and deliberately generous: on a thermally clamped
    machine wall time can be several times the compute time, and a timeout that fires
    because the CPU was being cooled would look exactly like a calculation that is too big.
    The recorded resources say which of the two actually happened.
    """
    rundir = workdir / system.key
    if rundir.exists():
        shutil.rmtree(rundir)
    rundir.mkdir(parents=True)
    inp = rundir / f"{system.key}.input"
    inp.write_text(build_input(system))
    env = dict(os.environ)
    env.update({"MOLCAS": str(MOLCAS_ROOT), "MOLCAS_NPROCS": str(nprocs),
                "MOLCAS_MEM": str(memory_mb), "MOLCAS_WORKDIR": str(rundir / "work"),
                "MOLCAS_PRINT": "2"})
    (rundir / "work").mkdir(exist_ok=True)
    tracker = thermal.track_resources()
    with tracker as res:
        status = _invoke(rundir, inp, env, timeout_s, system)
    return rundir, res, status


def _invoke(rundir: Path, inp: Path, env: Dict, timeout_s: int,
            system: sysdef.System) -> str:
    try:
        proc = subprocess.run([str(MOLCAS_ROOT / "pymolcas"), inp.name],
                              cwd=rundir, env=env, capture_output=True, text=True,
                              timeout=timeout_s)
        (rundir / f"{system.key}.out").write_text(proc.stdout + "\n" + proc.stderr)
        return "ok" if proc.returncode == 0 else f"pymolcas rc={proc.returncode}"
    except subprocess.TimeoutExpired:
        return f"timeout after {timeout_s} s wall"


# --- parsing --------------------------------------------------------------------------------
def read_aniso(path: Path) -> Dict[str, np.ndarray]:
    """Parse an OpenMolcas ``$Project.aniso`` (ANISOINPUT) file into ``{section: array}``.

    Format: a ``$name`` line, a line of dimensions, then the values in row-major order, five
    to a line, until the next section. This is OpenMolcas's own machine-readable interface to
    its property code — the same role ``props/dump.py`` plays for Kuiva — so it is the
    natural place to take the reference from. Note the ``rassi.h5`` datasets ``SOS_ANGMOM`` /
    ``SOS_SPIN`` are written as zeros unless extra RASSI keywords are given, whereas this
    file is always complete; taking the data from here avoids a silent all-zero reference.
    """
    sections: Dict[str, np.ndarray] = {}
    name: Optional[str] = None
    dims: Optional[Tuple[int, ...]] = None
    buf: List[float] = []

    def flush() -> None:
        if name is None or dims is None:
            return
        arr = np.asarray(buf, dtype=float)
        if arr.size == int(np.prod(dims)):
            sections[name] = arr.reshape(dims)

    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line.startswith("$"):
            flush()
            name = line[1:].split()[0] if len(line) > 1 else None
            dims, buf = None, []
            continue
        if name is None or not line:
            continue
        toks = line.split()
        if dims is None:
            try:
                dims = tuple(int(t) for t in toks)
            except ValueError:                       # a text section ($source, $atomlbl, ...)
                name = None
            continue
        try:
            buf.extend(float(t) for t in toks)
        except ValueError:
            name = None
    flush()
    return sections


def parse_aniso(path: Path) -> Dict:
    """Extract the SOC spectrum and the phase-invariant moment data from a ``.aniso`` file.

    Uses the magnetic-moment matrices OpenMolcas writes directly (``$magn_*``, in mu_B, the
    convention mu = -(L + g_e S)), rather than rebuilding them from L and S — fewer
    conventions to get wrong, and it is exactly the quantity the dump file holds.
    """
    sec = read_aniso(path)
    missing = [k for k in ("eso", "magn_xr", "magn_yr", "magn_zr") if k not in sec]
    if missing:
        raise KeyError(f"{path.name} is missing section(s) {missing}")
    e_sos = sec["eso"].ravel()
    e_sfs = sec["esfs"].ravel() if "esfs" in sec else None
    mu = np.array([sec[f"magn_{c}r"] + 1j * sec.get(f"magn_{c}i", 0.0) for c in "xyz"])
    mults = mult_mod.analyse_spectrum(e_sos, mu=mu, tol_cm=DEGENERACY_TOL_CM)

    out: Dict = {
        "n_soc_states": int(e_sos.size),
        "e_soc_total_lowest": float(np.min(e_sos)),
        "soc_rel_cm": [round(float(x), 4)
                       for x in (np.sort(e_sos) - np.min(e_sos)) * mult_mod.HARTREE_TO_CM],
        "degeneracy_pattern": list(mult_mod.degeneracy_pattern(mults)),
        "multiplets": [{"size": m.size,
                        "energy_cm": round(m.energy_cm, 4),
                        "spread_cm": round(m.spread_cm, 6),
                        "g_values": [round(g, 6) for g in m.g_values],
                        "m_tensor": [[round(float(v), 8) for v in row] for row in m.m_tensor]}
                       for m in mults],
    }
    if e_sfs is not None:
        out["n_spinfree_states"] = int(e_sfs.size)
        out["e_spinfree_lowest"] = float(np.min(e_sfs))
        out["spinfree_rel_cm"] = [
            round(float(x), 4)
            for x in (np.sort(e_sfs) - np.min(e_sfs)) * mult_mod.HARTREE_TO_CM]
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="", help="comma-separated system keys")
    ap.add_argument("--workdir", default="", help="scratch directory for the runs")
    ap.add_argument("--nprocs", type=int, default=1)
    ap.add_argument("--memory", type=int, default=3000, help="MOLCAS_MEM in MB")
    ap.add_argument("--timeout", type=int, default=7200, help="per-system timeout [s]")
    ap.add_argument("--merge", action="store_true",
                    help="merge into the existing reference file instead of replacing it")
    args = ap.parse_args(argv)

    if not molcas_available():
        print(f"OpenMolcas not found at {MOLCAS_ROOT}", file=sys.stderr)
        return 2
    workdir = Path(args.workdir) if args.workdir else REPO / "temp" / "tier2_molcas"
    workdir.mkdir(parents=True, exist_ok=True)

    out: Dict = {"schema": 2, "generator": "tests/generate/tier2_molcas.py",
                 "code": "OpenMolcas", "basis": BASIS,
                 "hamiltonian": "DKH2 (ANO-RCC) + AMFI spin-orbit, RASSCF/RASSI-SO",
                 "degeneracy_tol_cm": DEGENERACY_TOL_CM,
                 "environment": thermal.describe_environment(), "records": {}}
    if args.merge and REF_OUT.is_file():
        out = json.loads(REF_OUT.read_text())
        out.setdefault("records", {})

    keys = [k for k in args.only.split(",") if k] or None
    rc = 0
    for system in sysdef.for_tier2("molcas"):
        if keys and system.key not in keys:
            continue
        rundir, res, status = run_molcas(system, workdir, nprocs=args.nprocs,
                                         memory_mb=args.memory, timeout_s=args.timeout)
        rec: Dict = {"key": system.key, "label": system.label, "basis": BASIS,
                     "nroots": {str(k): v for k, v in system.nroots.items()},
                     "ncas": system.ncas, "nelecas": system.nelecas,
                     "resources": res.as_dict(), "status": status}
        # Judge success by whether a parseable .aniso was produced, not by pymolcas's exit
        # code: the trailing SINGLE_ANISO analysis may decline to run (non-Kramers systems)
        # long after RASSI has already written everything this reference needs.
        aniso = rundir / f"{system.key}.aniso"
        rec["pymolcas_status"] = status
        if aniso.is_file():
            try:
                rec.update(parse_aniso(aniso))
                status = rec["status"] = "ok"
                expected = system.soc_states
                if rec["n_soc_states"] != expected:
                    rec["status"] = (f"state-count mismatch: got {rec['n_soc_states']}, "
                                     f"expected {expected}")
            except Exception as exc:                            # noqa: BLE001
                rec["status"] = f"parse failed: {type(exc).__name__}: {exc}"
        else:
            rec["status"] = f"no .aniso file produced ({status})"
        out["records"][system.key] = rec
        if rec["status"] != "ok":
            rc = 1
        print(f"[molcas] {system.key:12s} {rec['status']:28s} "
              f"states={rec.get('n_soc_states', '-')} "
              f"pattern={rec.get('degeneracy_pattern', '-')} ({res.summary()})", flush=True)
        if res.throttled:
            # WARNING: completed, but the wall time above is cooling, not cost.
            print(f"  [warn] {system.key}: CPU thermally clamped for "
                  f"{100 * (res.throttle_fraction or 0):.0f}% of the run; do not read its "
                  f"wall time as the cost of the calculation", flush=True)

    REF_OUT.parent.mkdir(parents=True, exist_ok=True)
    REF_OUT.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"\nwrote {REF_OUT.relative_to(REPO)}  ({len(out['records'])} records)")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
