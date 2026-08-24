"""Example 2 -- spin-orbit coupling: the fine structure of a free atom.

    source setup.sh          # once per shell
    python atomic_spin_orbit.py

Runs in well under a minute on a boron atom and writes ``output/atomic_spin_orbit.out``.

WHAT THIS SHOWS
---------------
Spin-orbit coupling is the reason this program exists. It is not added to the CASSCF as a
correction afterwards: the Hamiltonian is two-component from the start, so the CI roots
*are* the spin-orbit eigenstates and their fine structure comes straight out of the
calculation.

Boron's ground configuration is 1s2 2s2 2p1. One electron in the six spinors of the 2p
shell is the smallest problem with a genuine fine structure, and it has answers no program
can argue with:

* the six states must split 2 + 4 -- a j = 1/2 level and a j = 3/2 level -- and each level
  must be exactly degenerate;
* every level of an odd-electron system is at least doubly degenerate (Kramers' theorem),
  which the four-fold j = 3/2 level satisfies twice over;
* the magnetic moment of each level is fixed by the Lande formula,
  g = 1 -/+ (g_e - 1)/3 for j = 1/2 and j = 3/2, with no free parameter anywhere.

The example computes the splitting twice -- with and without the two-electron part of the
spin-orbit operator -- because that is the single largest methodological choice in a
relativistic calculation of this kind, and it is on by default.

WHAT TO LOOK FOR IN THE OUTPUT
------------------------------
* the six-state spectrum, in the 2 + 4 pattern, at both settings;
* the splitting falling by roughly 40 per cent when the two-electron screening is included
  (that is the correction being large, not small);
* the principal g values of each level, matching Lande to a few units in the fourth
  decimal;
* the state average reported as *complete*: all six determinants are used, so no root was
  left out of it;
* <S^2> per level, and the term assignment built on it. Boron has one active electron, so
  <S^2> is 3/4 exactly and the two levels come out named 2P1/2 and 2P3/2 -- with L
  recovered from the measured g by inverting the Lande formula, which is a real check
  rather than a restatement. ⚠ Read the assignment table as what it says it is: every
  label in it is an INFERENCE from the block dimension, <S^2> and g, printed with its
  evidence and its fit residual, and withheld as "?" where they do not add up. It is a
  separate report and never a column of the state table, so nothing computed can be
  confused with something inferred.

⚠ WHAT <S^2> MEANS WITH SPIN-ORBIT COUPLING ON
-----------------------------------------------
It is not a quantum number. S stops being conserved the moment the Hamiltonian is
two-component, so 2S+1 is a measure of how pure the spin of a level still is, not a
multiplicity. It reads as a multiplicity here only because one electron cannot mix spins;
in a many-electron lanthanide it will not be integral, and how far it is from integral is
exactly how much a "6H15/2" label is worth. Every value is reported per degenerate BLOCK
and never per state, for the same reason the g values are: inside a block the eigensolver
may return any mixture of the members, so a single state's value belongs to that arbitrary
choice while the block trace does not.

READING THE COMPARISON WITH EXPERIMENT
--------------------------------------
The measured boron fine-structure splitting is 15.254 cm^-1 (NIST Atomic Spectra
Database). It is printed here as an anchor, not as an accuracy claim, and the residual
error is dominated by the small basis rather than by the Hamiltonian. Two rules bind any
spin-orbit splitting quoted anywhere:

* state which construction produced it -- a self-consistent two-component calculation and a
  frozen-orbital diagonalization of the same operator differ by tens of per cent. What is
  below is a state-averaged CASSCF, where the orbitals relax;
* never compare against a reference in a different basis. No picture-change correction can
  recover a basis truncation, and in a small basis the two errors partly cancel, so a
  strictly better Hamiltonian can move the total *away* from experiment. The number that
  says the two-electron correction is right is a four-component one in the same basis, and
  that comparison lives in the test suite, not here.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import kuiva
from kuiva.props.multiplet import G_ELECTRON, HARTREE_TO_CM
from kuiva.util import output as out
from kuiva.util import timing
from kuiva.util.logging import add_file_handler, get_logger

NAME = "atomic_spin_orbit"
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"

log = get_logger("examples." + NAME)

#: The 2p shell: six spinors holding one electron, and all six roots of it. Asking for
#: every root of the determinant space makes the state average complete by construction --
#: there is no root left outside it that could be degenerate with the last one taken.
N_ACTIVE, N_ACTIVE_ELEC, N_STATES = 6, 1, 6

#: Lande g factors of a p^1 term, with the real electron g factor rather than 2.
#: g = 1 + (g_e - 1) * [j(j+1) + s(s+1) - l(l+1)] / (2 j(j+1)), l = 1, s = 1/2.
G_LANDE = {2: 1.0 - (G_ELECTRON - 1.0) / 3.0,           # j = 1/2, two states
           4: 1.0 + (G_ELECTRON - 1.0) / 3.0}           # j = 3/2, four states

#: B I 2s^2 2p, 2P(3/2) - 2P(1/2), NIST Atomic Spectra Database. An anchor, not a target;
#: see the module docstring.
EXPERIMENT_CM = 15.254

#: A free atom is spherically symmetric, so the members of one j level are degenerate as a
#: matter of physics, not of numerics. 0.1 cm^-1 would already be a different physical
#: answer, so that is the bar -- a *physical* requirement, deliberately not set at whatever
#: spread this code happens to produce.
DEGENERACY_CM = 0.1

#: The two property routes below share states but not arithmetic; g values are compared at
#: 2e-3, which is above the residual anisotropy this small basis leaves (~1e-3) and far
#: below anything physical.
G_TOL = 2.0e-3

#: One active electron has S = 1/2 exactly -- there is no second spin for spin-orbit
#: coupling to mix it with -- so <S^2> = 3/4 is a *theorem* here, not a fitted tolerance.
#: 1e-8 is the numerical bar; the value comes out at 1e-15.
S2_TOL = 1.0e-8


def prepare_output() -> Path:
    """Write this run to output/<name>.out, with a scratch spin-orbit cache of its own."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cache = OUTPUT / "amf-cache"
    shutil.rmtree(cache, ignore_errors=True)
    os.environ["KUIVA_AMF_CACHE"] = str(cache)
    path = OUTPUT / (NAME + ".out")
    add_file_handler(path)
    return path


def fine_structure(screening: str) -> Tuple[object, object]:
    """A state-averaged two-component CASSCF on the boron 2p shell, and its moments.

    ``screening`` selects the spin-orbit Hamiltonian:

    ``"x2camf"``    the default. The one-electron X2C operator misses the screening of the
                    nucleus by the other electrons, which makes spin-orbit splittings 5 to
                    30 per cent too large. The atomic mean-field correction supplies it
                    from one four-component atomic calculation per element -- geometry
                    independent, so it is paid once ever and cached on disk.
    ``"none"``      no two-electron picture change. A statement about cost, not about
                    correctness: for a light element it is a fifth of a second, for a
                    lanthanide it is tens of minutes.
    """
    out.section(log, "Boron 2p, screening = {!r}".format(screening))

    boron = kuiva.Molecule(atoms=[("B", (0.0, 0.0, 0.0))], basis="x2c-SVPall-2c",
                           charge=0, spin=1)
    scf = kuiva.ScalarSCF(boron, memory_gb=2.0, screening=screening).run()
    ref = kuiva.Reference(scf).run()

    # The active space is stated as orbital CHARACTER, never as a window of orbital
    # indices: "the six lowest spinors of p character on the boron". An index window is not
    # a definition anyone else can reproduce, and it silently follows the basis set around.
    # Whole Kramers pairs always -- splitting a pair across a space boundary breaks the
    # conventions the CI addressing is built on.
    #
    # mode="second-order" is the right explicit choice when the orbital problem is hard
    # (a heavy element, a large state average); here the default "auto" is plenty.
    cas = kuiva.CASSCF(ref, character=("B", "p"), n_active=N_ACTIVE,
                       n_active_elec=N_ACTIVE_ELEC, n_states=N_STATES).run()

    # The property file: the effective Hamiltonian and the three magnetic-moment
    # components in the basis of the spin-orbit eigenstates. It is the program's actual
    # product -- an external crystal-field code reads it -- and example 6 is about the file
    # itself. Here it is simply the route to the g values.
    dump = kuiva.PropertyDump(cas, OUTPUT / "boron_{}.props".format(screening),
                              title="B 2p^1, CAS(1, 6), screening = " + screening).run()
    return cas, dump


def spectrum_table(title: str, energies: np.ndarray) -> None:
    out.subsection(log, title)
    table = out.Table(log, [
        out.col_count("state", 7),
        out.col_energy("E [Eh]"),
        out.Column("rel [cm^-1]", out.CM_FMT, 14),
    ])
    table.start()
    for i, energy in enumerate(energies):
        table.row(i, energy, (energies[i] - energies[0]) * HARTREE_TO_CM)
    table.end()


def level_table(dump) -> List[object]:
    """Report the degenerate levels of the spectrum and their principal g values.

    ⚠ The g values are read off a phase-INVARIANT reduction of the moment matrices, never
    off individual matrix elements. Nothing fixes a phase convention for degenerate states:
    they mix arbitrarily inside a level, so an element-by-element comparison of a moment
    matrix -- against another program, or against this program run twice -- is meaningless.
    Degeneracy patterns, relative energies and the invariant trace Tr(mu_i mu_j) are what
    can be compared.
    """
    levels = dump.matrices.analyse()
    table = out.Table(log, [
        out.col_count("level", 7),
        out.col_count("states", 8),
        out.Column("rel [cm^-1]", out.CM_FMT, 14),
        out.Column("spread [cm^-1]", out.SCI_FMT, 16),
        out.Column("g principal values", "{}", 28, align="<"),
        out.Column("Lande", "{}", 10, align="<"),
    ])
    table.start()
    for k, level in enumerate(levels):
        target = G_LANDE.get(level.size)
        table.row(k, level.size, level.energy_cm, level.spread_cm,
                  "  ".join("{:.5f}".format(g) for g in level.g_values),
                  "n/a" if target is None else "{:.5f}".format(target))
    table.end("j = 1/2 (2 states) and j = 3/2 (4 states); the pattern is a theorem")
    return levels


def main() -> int:
    outfile = prepare_output()
    out.banner(log, kuiva.__version__, "example 2: atomic fine structure (B 2p^1)")
    out.entry(log, "output file", str(outfile.relative_to(HERE)))

    results: Dict[str, Tuple[object, object]] = {}
    for screening in ("none", "x2camf"):
        results[screening] = fine_structure(screening)

    # ----------------------------------------------------------------------------------
    # The physics, side by side.
    # ----------------------------------------------------------------------------------
    out.section(log, "Fine structure")
    splittings: Dict[str, float] = {}
    levels: Dict[str, List[object]] = {}
    spreads: Dict[str, float] = {}
    for screening, (cas, dump) in results.items():
        energies = np.asarray(cas.energies, dtype=float)
        spectrum_table("boron 2p^1, screening = {!r}".format(screening), energies)
        levels[screening] = level_table(dump)
        # The splitting is between the two LEVELS, i.e. between their barycentres -- not
        # between two states picked out of them by index.
        splittings[screening] = (levels[screening][1].energy_cm
                                 - levels[screening][0].energy_cm)
        spreads[screening] = max(level.spread_cm for level in levels[screening])

    table = out.Table(log, [
        out.Column("spin-orbit Hamiltonian", "{}", 34, align="<"),
        out.Column("2P3/2 - 2P1/2 [cm^-1]", out.CM_FMT, 23),
        out.Column("vs experiment", "{:+.1f}%", 15),
    ])
    table.start()
    for screening, label in (("none", "X2C-1e (one-electron only)"),
                             ("x2camf", "X2C-AMF (default: + 2e screening)")):
        table.row(label, splittings[screening],
                  100.0 * (splittings[screening] / EXPERIMENT_CM - 1.0))
    table.end("experiment: {:.3f} cm^-1 (NIST). An anchor, not a target -- see the "
              "file header".format(EXPERIMENT_CM))

    # ----------------------------------------------------------------------------------
    # Naming the levels: <S^2> and the assignment offer.
    # ----------------------------------------------------------------------------------
    # Everything above identifies the two levels by their degeneracies, which is how it is
    # done by hand. Two measurements make the naming explicit.
    #
    # <S^2> is evaluated by applying S to the CI vectors. With spin-orbit coupling ON --
    # which it is here -- S is NOT a good quantum number, so 2S+1 is a measure of how pure
    # the spin still is rather than a multiplicity. Boron has one active electron and one
    # electron cannot mix spins, so it comes out at exactly 3/4 and the label 2S+1 = 2 is
    # safe. In a many-electron lanthanide it will not, and the number is then the thing
    # that says how much a "6H15/2" label is really worth.
    #
    # ⚠ It is reported per DEGENERATE BLOCK, never per state. Inside a degenerate block the
    # eigensolver may return any unitary mixture of the members, so an individual state's
    # <S^2> belongs to that arbitrary choice; the block trace does not. Same discipline as
    # the g values above, and for the same reason.
    #
    # The assignment then inverts the Lande formula for L, using the g value already
    # measured and the J the block dimension gives:
    #
    #     g_J = 3/2 + [S(S+1) - L(L+1)] / [2 J(J+1)]   ->   L(L+1)
    #
    # ⚠ Every label it prints is an INFERENCE, which is why it lives in its own report with
    # the evidence and a fit residual beside it, and never as a column of the state table.
    # A block whose evidence does not add up is labelled "?" rather than given the nearest
    # plausible term -- and that is the normal outcome for the crystal-field levels of a
    # complex, which are not 2J+1 manifolds at all. Here the free-ion picture is exact and
    # the residual is the ~1e-3 picture-change error of the bare L and S operators.
    out.section(log, "Naming the levels")
    assignment = results["x2camf"][1].assign()
    spin = results["x2camf"][0].spin_analysis()

    reduction = 100.0 * (1.0 - splittings["x2camf"] / splittings["none"])
    out.entries(log, [
        ("two-electron screening reduces the splitting by", reduction, "%", "", "{:.1f}"),
        ("largest spread inside a level (X2C-AMF)", spreads["x2camf"], "cm^-1",
         "must be zero to the physics", out.SCI_FMT),
        ("state average", "complete" if results["x2camf"][0].boundary.gap_cm is None
         else "{:.1f} cm^-1 to the next root".format(results["x2camf"][0].boundary.gap_cm),
         "", "all six determinants were used"),
    ])
    out.note(log, "the screening always makes the splitting smaller, and the structural")
    out.note(log, "statements -- the 2+4 pattern, the exact level degeneracies, the g")
    out.note(log, "values -- all survive it. A correction that broke one of them would be")
    out.note(log, "producing a plausible number for a broken reason.")

    # ----------------------------------------------------------------------------------
    # Assert. Everything here is a theorem or a closed-form value; nothing is a tolerance
    # fitted to what this code happens to print.
    # ----------------------------------------------------------------------------------
    checks = {}
    for screening in ("none", "x2camf"):
        tag = " ({})".format(screening)
        sizes = [level.size for level in levels[screening]]
        checks["the 2p shell splits into j = 1/2 + j = 3/2" + tag] = sizes == [2, 4]
        checks["each level is degenerate to better than {} cm^-1".format(DEGENERACY_CM)
               + tag] = spreads[screening] < DEGENERACY_CM
        checks["both levels carry their Lande g factor" + tag] = all(
            abs(g - G_LANDE[level.size]) < G_TOL
            for level in levels[screening] for g in level.g_values)
        checks["the CASSCF converged" + tag] = bool(results[screening][0].converged)
        checks["the state average uses the whole determinant space" + tag] = (
            results[screening][0].boundary.gap_cm is None)
    checks["two-electron screening reduces the splitting"] = (
        splittings["x2camf"] < splittings["none"])
    # One active electron is S = 1/2 exactly, whatever the spin-orbit coupling does to it:
    # there is no second spin to mix with. A value that is not 3/4 here would be a defect,
    # not a physical effect, which is what makes this assertable rather than a tolerance.
    checks["<S^2> = 3/4 on both levels (one electron cannot mix spins)"] = bool(
        np.max(np.abs(np.asarray(spin.block_s_squared) - 0.75)) < S2_TOL)
    # The assignment is an inference, so what is asserted is that the inference lands on
    # the level the theorems above already identified -- not that a label appeared.
    checks["the levels are assigned 2P1/2 and 2P3/2"] = (
        assignment.labels() == ("^2P_1/2", "^2P_3/2"))
    checks["the assignment recovers L = 1 from the Lande factor"] = all(
        term.orbital == 1 for term in assignment.terms)

    failures = report(checks)
    timing.summary(log)
    return 1 if failures else 0


def report(checks) -> int:
    """Print the check table and return the number of failures."""
    out.section(log, "Result")
    table = out.Table(log, [out.Column("check", "{}", 62, align="<"),
                            out.Column("verdict", "{}", 10, align="<")])
    table.start()
    for label, ok in checks.items():
        table.row(label, "ok" if ok else "FAILED")
    table.end()
    failed = [label for label, ok in checks.items() if not ok]
    for label in failed:
        log.error("check failed: %s", label)
    out.entry(log, "checks", "FAILED ({})".format(len(failed)) if failed
              else "all {} passed".format(len(checks)))
    return len(failed)


if __name__ == "__main__":
    raise SystemExit(main())
