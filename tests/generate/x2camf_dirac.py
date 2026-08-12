"""Four-component atomic references from DIRAC, for X2CAMF.

Why this file exists
--------------------
The existing Tier-2 cross-code tolerance is 15% and the error X2CAMF corrects is 16%
: **the Tier-2 suite cannot tell a working correction from a
broken one.** This generator builds the tighter reference the correction actually needs.

A four-component Dirac-Coulomb calculation has **no picture change at all** — it never
decouples anything — so reproducing its one-particle spectrum from a two-component
Hamiltonian *is* the statement that the picture-change treatment is correct. That makes it
the sharpest external check available, and it needs no new dependency: DIRAC 26.1 is already
installed .

⚠ What DIRAC contributes, and what it does not
----------------------------------------------
A four-component reference is **already available in-process and is free**:
``kuiva.amf.atomic.atomic_solution`` runs PySCF's ``scf.dhf`` on the same atom in the same
basis, and ``AtomicDiracSolution.occupied_energies()`` returns exactly the quantities parsed
here. **DIRAC's value is that it is a different program**, not that it is four-component.

So this generator is written to isolate precisely that. Every variable that is not "which
program" is pinned to what Kuiva does, and each one had to be pinned explicitly because
DIRAC's default differs from PySCF's in three of the four cases:

===================== ============================ ===============================
quantity              DIRAC default                pinned here to match Kuiva
===================== ============================ ===============================
nuclear model         Gaussian charge distribution ``.NUCMOD 1`` (point charge)
speed of light        CODATA 2022, 137.035999177   ``.CVALUE`` 137.035999084 (2018)
basis contraction     as given in the basis file   ``*READIN .UNCONTRACTED``
angular functions     spherical (l >= 2)           already matches PySCF
===================== ============================ ===============================

The nuclear model is the one that matters: it is a genuine physical difference that grows
with ``Z`` and would be read as a picture-change error. The ``c`` difference is 7e-10
relative and is pinned only because it costs one keyword to remove the question entirely.

Protocol
--------
* **Hamiltonian: four-component, both interactions.** ``.DOSSSS`` (Dirac-Coulomb *with* the
  ``(SS|SS)`` integral class) and the same plus ``.GAUNT``. Both are recorded per atom, and
  the **difference between them isolates the Gaunt (spin-other-orbit) screening** — one of
  the terms X2CAMF includes — which makes this a *decomposed* reference rather than a single
  number to compare against.

  ⚠ ``.DOSSSS`` is not optional here. Without it DIRAC replaces the ``(SS|SS)`` class by the
  Visscher point-charge correction, while PySCF's ``dhf.DHF`` computes it; the two would then
  differ by an approximation neither of them is being tested on. DIRAC's own X2CAMF tutorial
  makes the same point for its atomic runs.
* **Basis: dyall.v2z**, DIRAC-native and matched to PySCF's ``dyallv2z``. "Matched" means
  content-matched, not name-matched: the primitive exponents are cross-checked in
  ``tests/reference/basis_crosscheck.json`` (extended to cover the v2z atoms
  used here), which is the only thing that makes a cross-code number worth anything.
* **Closed-shell atoms only.** Open-shell ions need average-of-configuration DHF on the
  Kuiva side, which is the open-shell path; generating DIRAC records for them before there is
  anything to compare them against would be reference data with no consumer.

⚠ Cost, measured — do not assume DIRAC is the slow half
--------------------------------------------------------
Kuiva's *own* uncontracted atomic four-component solve grows as roughly ``n4c^4``: 0.2 s
wall for Ne but 177 s for Xe in ``x2c-SVPall-2c``. Both sides have to be
produced, and on that evidence the in-house side is the one to bound first. Every run here
is therefore bounded explicitly and **every record is written as soon as it exists**, so a
generator killed at its budget still leaves the atoms it finished .

**Rn is deliberately not in the atom set.** Its ``dyallv2z`` basis is 210 scalar functions
against Xe's 121, i.e. ~9x the four-component cost of an atom that already takes minutes, so
it is over the ten-minute line on the *Kuiva* side before DIRAC is even started. It
would need an explicit decision to spend, not a default.

References
----------
* DIRAC: T. Saue et al., J. Chem. Phys. 152, 204104 (2020), doi:10.1063/5.0004844;
  DIRAC26 (2026), http://www.diracprogram.org, doi:10.5281/zenodo.3572669.
* Four-component Dirac-Hartree-Fock: I. P. Grant, "Relativistic Quantum Theory of Atoms and
  Molecules", Springer (2007); K. G. Dyall, K. Faegri, "Introduction to Relativistic
  Quantum Chemistry", Oxford University Press (2007), ch. 7 and 11.
* The Gaunt interaction: J. A. Gaunt, Proc. R. Soc. Lond. A 122, 513 (1929).
* The ``(SS|SS)`` point-charge correction DIRAC applies by default (and which ``.DOSSSS``
  turns off): L. Visscher, Theor. Chem. Acc. 98, 68 (1997), doi:10.1007/s002140050280.
* X2CAMF: J. Liu, L. Cheng, J. Chem. Phys. 148, 144108 (2018), doi:10.1063/1.5023750.
* DIRAC's own (e)amfX2C implementation, whose atomic-run protocol this follows:
  S. Knecht, M. Repisky, H. J. Aa. Jensen, T. Saue, J. Chem. Phys. 157, 114106 (2022),
  doi:10.1063/5.0095112.
* Dyall basis sets: K. G. Dyall, Theor. Chem. Acc. 135, 128 (2016) and companion papers;
  http://dirac.chem.sdu.dk.

Run:  python tests/generate/x2camf_dirac.py [--only Ne,Ar] [--hamiltonian dc,dcg]
(with ``external/env.sh`` sourced). Writes ``tests/reference/x2camf_dirac.json``.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import thermal                                                # noqa: E402

REF_OUT = REPO / "tests/reference/x2camf_dirac.json"
DIRAC_ROOT = REPO / "external/install/dirac"
PAM = DIRAC_ROOT / "share/dirac/pam"

#: DIRAC's name for the basis; PySCF calls the same set ``dyallv2z``. Content-matched, see
#: the module docstring.
BASIS = "dyall.v2z"
PYSCF_BASIS = "dyallv2z"

#: The speed of light [a.u.] PySCF uses (``pyscf.lib.param.LIGHT_SPEED``, CODATA 2018), forced
#: on DIRAC so that ``c`` is not a variable in the comparison. DIRAC 26.1 defaults to CODATA
#: 2022 (137.035999177).
LIGHT_SPEED = 137.035999084

HARTREE_TO_CM = 219474.6313632

#: The two four-component interactions recorded per atom. Their difference isolates the Gaunt
#: (spin-other-orbit) screening — see the module docstring.
HAMILTONIANS = ("dc", "dcg")


@dataclass(frozen=True)
class AtomSpec:
    """One atom or ion: its shell structure and the valence shell under test.

    Attributes
    ----------
    symbol, z : str, int
    closed_g, closed_u : int
        **Closed-shell** electron counts per fermion irrep (gerade, ungerade). DIRAC fills the
        lowest spinors of each irrep and does **not** check the split against the electron
        configuration, so a wrong split converges silently to the wrong state — the trap
        recorded at length in ``tier2_dirac.py``. Counted here from the configuration: gerade
        = all s and d shells, ungerade = all p shells. For an open-shell species this is the
        **core only**; the frontier shell goes in ``open_*`` below.
    valence : str
        Label of the valence shell, for the record. Provenance, not used in any comparison.
    valence_spinors : int
        Spinors spanned by the valence shell — 6 for a ``p`` shell (``p_1/2`` 2 + ``p_3/2``
        4), 10 for ``d``, 14 for ``f`` — so the spread of those occupied spinor energies **is**
        the one-particle spin-orbit splitting. Matches
        ``AtomicDiracSolution.shell_splitting()``.
    charge : int
        Net charge. ⚠ Derived from nothing: it must agree with ``configuration``, and
        :func:`_check_consistency` asserts that it does rather than trusting the table.
    configuration : str
        The reference configuration in Kuiva's notation, e.g. ``"[Ar]3d1"``. This is what the
        Kuiva side of the comparison is run in, so it lives here rather than in the test —
        one source of truth, the same discipline ``tests/generate/systems.py`` applies.
    open_electrons, open_g, open_u : int
        Average-of-configuration open shell: how many electrons are distributed over how many
        **spinors** in each fermion irrep. ``open_electrons == 0`` means a closed shell and no
        ``.OPEN SHELL`` block is written at all, so every closed-shell record generated before
        the open-shell record is reproduced byte for byte.

        ⚠ **The open shell must span the whole ``nl`` shell, not a ``j`` sub-shell.** Bi 6p³
        is 3 electrons over 6 spinors, *not* 2 over ``6p_1/2`` plus 1 over ``6p_3/2``. Both
        occupations are spherical, so no symmetry check separates them, and Kuiva's own
        occupation rule (``kuiva.amf.configuration.average_occupations``) makes the same
        choice — asserted on the *value* of the
        fraction for exactly that reason.
    """

    symbol: str
    z: int
    closed_g: int
    closed_u: int
    valence: str
    valence_spinors: int = 6
    charge: int = 0
    configuration: str = ""
    open_electrons: int = 0
    open_g: int = 0
    open_u: int = 0

    @property
    def n_electrons(self) -> int:
        return self.z - self.charge

    @property
    def is_open_shell(self) -> bool:
        return self.open_electrons > 0


#: The atom set. ⚠ ``Rn`` is deliberately absent — see the module docstring — and the
#: open-shell ions are the expensive half: their **Kuiva-side** four-component
#: average-of-configuration solve is 15-60 minutes each , so they are
#: `slow`-marked in the test and are run by name here.
ATOMS: Tuple[AtomSpec, ...] = (
    # --- closed shell: the closed-shell set, unchanged --------------------------------------
    # Ne 1s2 2s2 2p6:            g = 1s,2s = 4      u = 2p = 6
    AtomSpec("Ne", 10, 4, 6, "2p", configuration="[He]2s2 2p6"),
    # Ar [Ne] 3s2 3p6:           g = 1s,2s,3s = 6   u = 2p,3p = 12
    AtomSpec("Ar", 18, 6, 12, "3p", configuration="[Ne]3s2 3p6"),
    # Kr [Ar] 3d10 4s2 4p6:      g = 4 s + 3d = 18  u = 2p,3p,4p = 18
    AtomSpec("Kr", 36, 18, 18, "4p", configuration="[Ar]3d10 4s2 4p6"),
    # Xe [Kr] 4d10 5s2 5p6:      g = 5 s + 3d,4d = 30   u = 2p..5p = 24
    AtomSpec("Xe", 54, 30, 24, "5p", configuration="[Kr]4d10 5s2 5p6"),

    # --- open shell ------------------------------------------------------
    # ⚠ C and O are here for a reason that is not their chemistry: they are the **cheap**
    # open-shell cases (seconds on both sides), and the ratio of their two occupation
    # fractions is what distinguishes true average-of-configuration from plain
    # fractional-occupation Hartree-Fock. The open-shell two-electron energy carries
    # ``q(q-1)/(n(n-1))`` in AOC and ``(q/n)^2`` under a fractional density; the two differ
    # most at small ``q`` and agree as the shell fills. C (2p², 0.067 vs 0.111) and O (2p⁴,
    # 0.20 vs 0.44 relative gap) bracket that without spending a lanthanide on it.
    # C 1s2 2s2 2p2: g = 1s,2s = 4; open = 2 electrons over the six 2p spinors (ungerade).
    AtomSpec("C", 6, 4, 0, "2p", valence_spinors=6, charge=0,
             configuration="[He]2s2 2p2", open_electrons=2, open_g=0, open_u=6),
    # O 1s2 2s2 2p4
    AtomSpec("O", 8, 4, 0, "2p", valence_spinors=6, charge=0,
             configuration="[He]2s2 2p4", open_electrons=4, open_g=0, open_u=6),
    # ⚠ **Parity is ``(-1)^l``, so ``s`` and ``d`` are gerade while ``p`` and ``f`` are
    # UNGERADE.** A ``d`` open shell goes in ``open_g`` and a ``p`` *or* ``f`` one in
    # ``open_u``. Getting it backwards does not fail: DIRAC puts the electrons in the wrong
    # fermion irrep, converges cleanly, and reports a "4f splitting" measured over whatever
    # gerade spinors happened to be next — measured, Ce(3+) came back at **116076 cm^-1** and
    # Yb(3+) at 369470 for a splitting of order 2000. The magnitude is the only thing that
    # gave it away, which is exactly why the parity is spelled out here.
    # Ti(3+) [Ar]3d1: core g = 1s,2s,3s = 6 + 2p,3p ... (p is ungerade) -> g = 6, u = 12;
    #                 open = 1 electron over the ten 3d spinors, all gerade.
    AtomSpec("Ti", 22, 6, 12, "3d", valence_spinors=10, charge=3,
             configuration="[Ar]3d1", open_electrons=1, open_g=10, open_u=0),
    # Ce(3+) [Xe]4f1: core = Xe (g 30, u 24); open = 1 over the fourteen 4f spinors (gerade).
    AtomSpec("Ce", 58, 30, 24, "4f", valence_spinors=14, charge=3,
             configuration="[Xe]4f1", open_electrons=1, open_g=0, open_u=14),
    # Dy(3+) [Xe]4f9
    AtomSpec("Dy", 66, 30, 24, "4f", valence_spinors=14, charge=3,
             configuration="[Xe]4f9", open_electrons=9, open_g=0, open_u=14),
    # Yb(3+) [Xe]4f13 — the one-hole partner of Ce(3+)'s one electron.
    AtomSpec("Yb", 70, 30, 24, "4f", valence_spinors=14, charge=3,
             configuration="[Xe]4f13", open_electrons=13, open_g=0, open_u=14),
    # Bi [Xe]4f14 5d10 6s2 6p3 — ⚠ **4f is ungerade**, which is what makes this row easy to
    # get wrong (it was, and the parity check in parse_output is what caught it):
    #   g = 1s..6s (12) + 3d,4d,5d (30)               = 42
    #   u = 2p,3p,4p,5p (24) + 4f (14)                = 38 ... 6p is the open shell.
    AtomSpec("Bi", 83, 42, 38, "6p", valence_spinors=6, charge=0,
             configuration="[Xe]4f14 5d10 6s2 6p3", open_electrons=3, open_g=0, open_u=6),
)


def _check_consistency(atom: AtomSpec) -> None:
    """⚠ The electron count must come out of the shell structure, not be asserted beside it.

    ``closed_g + closed_u + open_electrons`` is what DIRAC will actually put in the atom, and
    ``z - charge`` is what the configuration says it should be. A mismatch is the failure mode
    ``tier2_dirac.py`` documents at length: DIRAC fills the lowest spinors per irrep without
    checking, converges cleanly, and leaves a shell short. Checked here so it cannot reach a
    committed record.
    """
    filled = atom.closed_g + atom.closed_u + atom.open_electrons
    if filled != atom.n_electrons:
        raise ValueError(
            "{}{:+d}: the shell structure holds {} electrons ({} g + {} u closed, {} open) "
            "but the charge implies {}. DIRAC would fill the lowest spinors per irrep without "
            "complaining and converge to the wrong state.".format(
                atom.symbol, atom.charge, filled, atom.closed_g, atom.closed_u,
                atom.open_electrons, atom.n_electrons))
    if atom.is_open_shell and atom.open_g + atom.open_u != atom.valence_spinors:
        raise ValueError(
            "{}{:+d}: the open shell spans {} spinors but the valence shell is {} wide. The "
            "average must cover the whole nl shell, not a j sub-shell.".format(atom.symbol, atom.charge, atom.open_g + atom.open_u,
                              atom.valence_spinors))


def dirac_available() -> bool:
    return PAM.is_file()


# --- input construction -------------------------------------------------------------------
def build_mol(atom: AtomSpec) -> str:
    """A DIRAC ``.mol`` file (MOLECULE format) for one atom at the origin."""
    return ("INTGRL\n"
            "{} atom, four-component reference for X2CAMF\n"
            "{}\n"
            "C   1              A\n"
            "{:10.1f} {:4d}\n"
            "{:<6s}{:16.10f}{:16.10f}{:16.10f}\n"
            "LARGE BASIS {}\n"
            "FINISH\n").format(atom.symbol, BASIS, float(atom.z), 1, atom.symbol,
                               0.0, 0.0, 0.0, BASIS)


def build_inp(atom: AtomSpec, hamiltonian: str) -> str:
    """A DIRAC input: closed-shell four-component SCF, Coulomb or Coulomb+Gaunt.

    Every non-default keyword here removes a variable from the comparison rather than
    changing the physics being compared; see the table in the module docstring.
    """
    if hamiltonian not in HAMILTONIANS:
        raise ValueError("unknown hamiltonian {!r}; expected one of {}".format(
            hamiltonian, HAMILTONIANS))
    _check_consistency(atom)
    gaunt = "\n.GAUNT" if hamiltonian == "dcg" else ""
    # ⚠ Written only for an open shell, so every closed-shell input is byte-identical to the
    # the closed-shell one, so the committed closed-shell records stay valid.
    # ``.OPEN SHELL`` is `n_shells / newline / electrons/spinors_g,spinors_u`, i.e. DIRAC's
    # average-of-configuration: the electrons are spread evenly over the listed spinors and
    # the density stays spherical. This is the same state Kuiva's
    # ``configuration.average_occupations`` produces, which is what makes the two comparable.
    open_block = ""
    if atom.is_open_shell:
        open_block = "\n.OPEN SHELL\n 1\n {}/{},{}".format(
            atom.open_electrons, atom.open_g, atom.open_u)
    # An AOC SCF needs more cycles than a closed shell; 80 is DIRAC's own default here and is
    # not enough for a lanthanide.
    maxitr = 200 if atom.is_open_shell else 80
    return """**DIRAC
.TITLE
 {sym} four-component {ham} atomic SCF, reference for X2CAMF
.WAVE FUNCTION
**GENERAL
.CVALUE
 {c:.9f}d0
**INTEGRALS
.NUCMOD
 1
*READIN
.UNCONTRACTED
**HAMILTONIAN
.DOSSSS{gaunt}
**WAVE FUNCTION
.SCF
*SCF
.CLOSED SHELL
 {cg} {cu}{open_block}
.MAXITR
 {maxitr}
*END OF
""".format(sym=atom.symbol, ham=hamiltonian.upper(), c=LIGHT_SPEED, gaunt=gaunt,
           cg=atom.closed_g, cu=atom.closed_u, open_block=open_block, maxitr=maxitr)


# --- running ------------------------------------------------------------------------------
def run_dirac(atom: AtomSpec, hamiltonian: str, workdir: Path, timeout_s: int,
              memory_gb: float) -> Tuple[Path, "thermal.RunResources", str]:
    """Run one atom with one Hamiltonian; returns (output path, resources, status)."""
    rundir = workdir / "{}_{}".format(atom.symbol.lower(), hamiltonian)
    if rundir.exists():
        shutil.rmtree(rundir)
    rundir.mkdir(parents=True)
    molname = "{}.mol".format(atom.symbol.lower())
    inpname = "{}.inp".format(hamiltonian)
    (rundir / molname).write_text(build_mol(atom))
    (rundir / inpname).write_text(build_inp(atom, hamiltonian))
    with thermal.track_resources() as res:
        try:
            proc = subprocess.run(
                [str(PAM), "--noarch", "--gb={}".format(memory_gb),
                 "--inp={}".format(inpname), "--mol={}".format(molname)],
                cwd=rundir, env=dict(os.environ), capture_output=True, text=True,
                timeout=timeout_s)
            (rundir / "pam.log").write_text(proc.stdout + "\n" + proc.stderr)
            status = "ok" if proc.returncode == 0 else "pam rc={}".format(proc.returncode)
        except subprocess.TimeoutExpired:
            status = "timeout after {} s wall".format(timeout_s)
    out_file = rundir / "{}_{}.out".format(hamiltonian, atom.symbol.lower())
    return out_file, res, status


#: Lines worth keeping when a run fails, so a record says *why* an atom has no data rather
#: than silently omitting it (the convention ``tier2_dirac.py`` established).
_ERROR_PATTERNS = ("Fatal memory allocation error", "MEMGET ERROR", "SELF CONSISTENCE",
                   "TOO MANY SCF ITERATIONS", "input error", "does not match",
                   "SIGSEGV", "FATAL ERROR")


def dirac_error(out_path: Path) -> str:
    """The most informative error line from a failed DIRAC run (for the record)."""
    try:
        text = out_path.read_text(errors="replace")
    except OSError:
        return "output unreadable"
    hits = [ln.strip() for ln in text.splitlines() if any(p in ln for p in _ERROR_PATTERNS)]
    return hits[-1] if hits else "no recognised error message"


# --- parsing ------------------------------------------------------------------------------
_BLOCK = re.compile(r"^\s*\*\s*Block\s+\d+\s+in\s+(\S+):\s*(.+?)\s*$")
#: ⚠ DIRAC numbers its open shells — ``* Open shell #1, f = 0.1000`` — but not its closed one
#: (``* Closed shell, f = 1.0000``). A pattern written for the closed-shell form alone matches
#: nothing in an average-of-configuration run, and the parser then sees only the closed core:
#: measured on Ti(3+), 18.0 electrons where 19 were expected. It failed loudly because the
#: electron count is checked, which is the whole reason that check is separate from the
#: spinor count.
_OCCUPIED = re.compile(r"^\s*\*\s*(Closed|Open) shell(?:\s*#\d+)?,\s*f\s*=\s*([\d.]+)")
_VIRTUAL = re.compile(r"^\s*\*\s*Virtual eigenvalues")
_EIGENVALUE = re.compile(r"(-?\d+\.\d+)\s*\(\s*(\d+)\)")
_TOTAL_ENERGY = re.compile(r"^\s*Total energy\s*:\s*(-?\d+\.\d+)", re.MULTILINE)
_SPEED_OF_LIGHT = re.compile(r"speed of light scaled by.*to\s*:\s*(\d+\.\d+)")
_NUCLEAR_MODEL = re.compile(r"Nuclear model requested in input:\s*(.+?)\.")


def parse_output(out_path: Path, atom: AtomSpec) -> Dict:
    """Extract the occupied four-component spinor spectrum from a DIRAC output.

    DIRAC prints its ``Eigenvalues`` section one *block* at a time — one block per
    ``(shell, m_j)`` pair within a fermion irrep — each energy followed by its degeneracy in
    parentheses (``( 2)`` for a Kramers pair). Occupied entries sit under
    ``* Closed shell, f = 1.0000``; virtuals under ``* Virtual eigenvalues``.

    ⚠ **The blocks must not be de-duplicated.** ``p 3/2; 1/2`` and ``p 3/2; -3/2`` carry the
    same energy by symmetry and are different states; collapsing them would silently halve the
    ``p_3/2`` degeneracy and turn a correct spectrum into a plausible wrong one — the same
    trap ``tier2_dirac.py`` records for its Omega blocks.

    Open shells (``* Open shell, f = 0.xxxx``) carry a fractional occupation, so the electron
    count and the spinor count stop being the same number and are checked **separately** —
    see below for why only both together pin the state.
    """
    text = out_path.read_text(errors="replace")
    shells: List[Dict] = []
    mode: Optional[str] = None
    label = ""
    occupation = 0.0
    for line in text.splitlines():
        m = _BLOCK.match(line)
        if m:
            label, mode = m.group(2), None
            continue
        m = _OCCUPIED.match(line)
        if m:
            mode, occupation = "occupied", float(m.group(2))
            continue
        if _VIRTUAL.match(line) or line.strip().startswith("* Occupation in fermion"):
            mode = None
            continue
        if mode != "occupied":
            continue
        for energy, degeneracy in _EIGENVALUE.findall(line):
            shells.append({"block": label, "shell": label.split(";")[0].strip(),
                           "energy": float(energy), "degeneracy": int(degeneracy),
                           "occupation": occupation})
    if not shells:
        raise ValueError("no occupied eigenvalues found in {}".format(out_path.name))

    # (energy, occupation) per spinor, so the two checks below can be different questions.
    occupied: List[Tuple[float, float]] = []
    for s in shells:
        occupied.extend([(s["energy"], s["occupation"])] * s["degeneracy"])
    occupied.sort()
    spinors = [e for e, _ in occupied]

    # ⚠ Two independent checks, because for an open shell they say different things and only
    # together do they pin the state. The **electron count** is what fixes the charge; the
    # **spinor count** is what fixes how many spinors the average was spread over — and an
    # average over a j sub-shell instead of the whole nl shell has the right electron count,
    # a spherical density, and the wrong answer.
    n_electrons = sum(occ for _, occ in occupied)
    # ⚠ The occupation is **read from a printed number**: DIRAC writes ``f = 0.3333`` to four
    # decimals, so 6 spinors of a 2p² shell sum to 5.9998 rather than 6. The tolerance is the
    # accumulated rounding of that printout, not a physical one — a genuinely wrong split is
    # off by a whole electron, three orders above this.
    if abs(n_electrons - atom.n_electrons) > 1e-4 * max(len(occupied), 1):
        raise ValueError(
            "{}{:+d}: the occupied spinors hold {:.4f} electrons, expected {} — the "
            ".CLOSED SHELL / .OPEN SHELL split does not match the configuration".format(
                atom.symbol, atom.charge, n_electrons, atom.n_electrons))
    n_expected = atom.closed_g + atom.closed_u + atom.open_g + atom.open_u
    if len(spinors) != n_expected:
        raise ValueError(
            "{}{:+d}: got {} occupied spinors, expected {} ({} closed + {} open)".format(
                atom.symbol, atom.charge, len(spinors), n_expected,
                atom.closed_g + atom.closed_u, atom.open_g + atom.open_u))
    if atom.is_open_shell:
        fractional = {round(occ, 6) for _, occ in occupied if occ < 1.0 - 1e-6}
        expected_f = atom.open_electrons / float(atom.open_g + atom.open_u)
        if len(fractional) != 1 or abs(list(fractional)[0] - expected_f) > 1e-3:
            raise ValueError(
                "{}{:+d}: the open shell came back with fractional occupations {} where a "
                "single value of {:.4f} was expected. DIRAC averaged over a different set of "
                "spinors from the one requested.".format(atom.symbol, atom.charge,
                                                         sorted(fractional), expected_f))

    totals = _TOTAL_ENERGY.findall(text)
    c = _SPEED_OF_LIGHT.findall(text)
    nucmod = _NUCLEAR_MODEL.findall(text)
    # ⚠ **The valence manifold of an open-shell atom must not be taken by index**, and this is
    # the ``"6p"`` AO-label error in a new place. "The top ``4l+2`` occupied spinors"
    # presumes the frontier shell is highest in energy — true for a closed-shell p-block atom,
    # and false for a lanthanide, where 4f sits *below* 5s and 5p. Taken by index, Ce(3+)'s
    # 4f¹ "splitting" came out at **116076 cm⁻¹** and Dy(3+)'s at 252283: a window straddling
    # 4f, 5s and 5p, reported as spin-orbit coupling. the same trap is recorded at the mean field's own configuration rules —
    # trap on the Kuiva side, where it is solved by assigning spinors to an ``l`` channel.
    #
    # Here the answer is exact and needs no assignment at all: the open shell **is** the
    # valence shell, and DIRAC labels it by its fractional occupation. For a closed-shell atom
    # there is no open shell and the index window is correct — and is kept, so every record
    # committed before this reparses to the same number.
    if atom.is_open_shell:
        open_entries = [s for s in shells if s["occupation"] < 1.0 - 1e-3]
        valence = sorted(e for e, occ in occupied if occ < 1.0 - 1e-3)
        if len(valence) != atom.valence_spinors:
            raise ValueError(
                "{}{:+d}: the fractionally occupied set has {} spinors where the {} shell "
                "spans {}".format(atom.symbol, atom.charge, len(valence), atom.valence,
                                  atom.valence_spinors))
        # ⚠ **And the open spinors must actually be the shell they are supposed to be.**
        # DIRAC labels each block by angular momentum (``d 5/2; -3/2``), so this is exact and
        # free. It exists because ``.OPEN SHELL`` takes per-**fermion-irrep** counts and
        # parity is ``(-1)^l``: putting a 4f shell in the gerade column is a one-character
        # error, DIRAC converges cleanly on whatever gerade spinors were next, and the only
        # symptom is a "4f splitting" of 116076 cm^-1. Measured, and it reached a committed
        # record before this check existed.
        letter = atom.valence[-1]
        labels = sorted({e["shell"].split()[0] for e in open_entries})
        if labels != [letter]:
            raise ValueError(
                "{}{:+d}: the open shell came back on {} spinors, not {} — check the parity "
                "of the .OPEN SHELL irrep split (s, d gerade; p, f ungerade)".format(
                    atom.symbol, atom.charge, "/".join(labels), letter))
    else:
        valence = spinors[-atom.valence_spinors:]
    return {
        "e_total": float(totals[-1]) if totals else None,
        "n_occupied_spinors": len(spinors),
        "n_electrons": float(n_electrons),
        "spinor_energies": [float(x) for x in spinors],
        "spinor_occupations": [float(o) for _, o in occupied],
        "shells": [dict(s) for s in shells],
        "valence_shell": atom.valence,
        "valence_spinors": atom.valence_spinors,
        "valence_splitting_eh": float(valence[-1] - valence[0]),
        "valence_splitting_cm": float((valence[-1] - valence[0]) * HARTREE_TO_CM),
        "speed_of_light": float(c[-1]) if c else None,
        "nuclear_model": nucmod[-1].strip().lower() if nucmod else None,
    }


# --- driver -------------------------------------------------------------------------------
def _new_document() -> Dict:
    return {
        "schema": 1,
        "generator": "tests/generate/x2camf_dirac.py",
        "code": "DIRAC 26.1",
        "purpose": ("four-component atomic references for the X2CAMF two-electron "
                    "picture-change correction"),
        "basis": BASIS,
        "pyscf_basis": PYSCF_BASIS,
        "controlled": {
            "nuclear_model": "point charge (.NUCMOD 1; DIRAC defaults to Gaussian)",
            "speed_of_light": LIGHT_SPEED,
            "speed_of_light_note": ("forced to PySCF's CODATA-2018 value; DIRAC 26.1 "
                                    "defaults to CODATA 2022, 137.035999177"),
            "contraction": "uncontracted (*READIN .UNCONTRACTED), matching uncontract=True",
            "two_electron": "(SS|SS) computed explicitly (.DOSSSS), not Visscher-corrected",
        },
        "hamiltonians": {
            "dc": "four-component Dirac-Coulomb with (SS|SS)",
            "dcg": "four-component Dirac-Coulomb-Gaunt with (SS|SS)",
        },
        "environment": thermal.describe_environment(),
        "records": {},
    }


def _write(document: Dict) -> None:
    """Write the document. Called after **every** record (a run killed at
    its budget must still yield what it finished)."""
    REF_OUT.parent.mkdir(parents=True, exist_ok=True)
    REF_OUT.write_text(json.dumps(document, indent=2, sort_keys=True))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="", help="comma-separated element symbols")
    ap.add_argument("--hamiltonian", default=",".join(HAMILTONIANS),
                    help="comma-separated subset of {}".format(HAMILTONIANS))
    ap.add_argument("--workdir", default="")
    ap.add_argument("--timeout", type=int, default=900,
                    help="wall-clock bound per DIRAC run [s]")
    ap.add_argument("--max-wall", type=float, default=3000.0,
                    help="total wall budget [s]; no new run is started past it")
    ap.add_argument("--memory-gb", type=float, default=4.0, help="DIRAC WORK array size")
    ap.add_argument("--fresh", action="store_true",
                    help="discard existing records instead of merging into them")
    ap.add_argument("--reparse", action="store_true",
                    help="re-parse the DIRAC outputs already in the workdir instead of "
                         "re-running DIRAC. ⚠ Exists because a *parser* fix must not cost "
                         "hours of four-component SCF: the lanthanide runs are ~10 minutes "
                         "each and the valence-window bug that made this necessary was in "
                         "the parser, not in DIRAC.")
    args = ap.parse_args(argv)

    if not dirac_available():
        print("DIRAC not found at {}".format(PAM), file=sys.stderr)
        return 2
    workdir = Path(args.workdir) if args.workdir else REPO / "temp" / "x2camf_dirac"
    workdir.mkdir(parents=True, exist_ok=True)

    document = _new_document()
    if not args.fresh and REF_OUT.is_file():
        try:
            existing = json.loads(REF_OUT.read_text())
            document["records"] = existing.get("records", {})
        except ValueError:
            pass                                  # corrupt file: start over rather than fail

    wanted = {s.strip().capitalize() for s in args.only.split(",") if s.strip()}
    hams = [h.strip() for h in args.hamiltonian.split(",") if h.strip()]
    started = time.time()
    rc = 0
    for atom in ATOMS:
        if wanted and atom.symbol not in wanted:
            continue
        for ham in hams:
            elapsed = time.time() - started
            if elapsed > args.max_wall:
                print("[budget] {:.0f} s of {:.0f} s used; stopping before {} {}".format(
                    elapsed, args.max_wall, atom.symbol, ham), flush=True)
                return rc
            key = "{}/{}".format(atom.symbol, ham)
            if args.reparse:
                out_file = (workdir / "{}_{}".format(atom.symbol.lower(), ham)
                            / "{}_{}.out".format(ham, atom.symbol.lower()))
                status = "reparsed"
                res = thermal.RunResources()
                previous = document["records"].get(key, {})
            else:
                out_file, res, status = run_dirac(atom, ham, workdir, args.timeout,
                                                 args.memory_gb)
                previous = {}
            record: Dict = {"element": atom.symbol, "atomic_number": atom.z,
                            "charge": atom.charge, "configuration": atom.configuration,
                            "open_shell": atom.is_open_shell,
                            "open_electrons": atom.open_electrons,
                            "open_spinors": atom.open_g + atom.open_u,
                            "basis": BASIS, "hamiltonian": ham, "uncontracted": True,
                            "resources": previous.get("resources", res.as_dict()),
                            "status": status}
            if out_file.is_file():
                try:
                    record.update(parse_output(out_file, atom))
                    record["status"] = "ok"
                except Exception as exc:                             # noqa: BLE001
                    record["status"] = "parse failed: {}: {}".format(type(exc).__name__, exc)
                    record["diagnostic"] = dirac_error(out_file)
            else:
                record["status"] = "no output file ({})".format(status)
            document["records"][key] = record
            _write(document)                       # incremental, within the ten-minute ad-hoc budget
            if record["status"] != "ok":
                rc = 1
            print("[dirac] {:8s} {:24s} E = {:>18}  {} splitting = {:>9} cm^-1  ({})".format(
                key, record["status"],
                "{:.8f}".format(record["e_total"]) if record.get("e_total") else "-",
                atom.valence,
                "{:.2f}".format(record["valence_splitting_cm"])
                if record.get("valence_splitting_cm") is not None else "-",
                res.summary()), flush=True)
            if res.throttled:
                # WARNING: completed, but the wall time is cooling, not cost.
                print("  [warn] {}: CPU thermally clamped for {:.0f}% of the run".format(
                    key, 100 * (res.throttle_fraction or 0)), flush=True)

    _write(document)
    print("\nwrote {}  ({} records)".format(REF_OUT.relative_to(REPO),
                                            len(document["records"])))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
