"""Example 5 -- dynamic correlation: SC-NEVPT2 on a converged two-component CASSCF.

    source setup.sh          # once per shell
    python nevpt2.py

Runs in a few seconds on an oxygen atom and writes ``output/nevpt2.out``.

WHAT THIS SHOWS
---------------
A CASSCF describes the strong, near-degenerate correlation inside the active space and
nothing else. Everything outside it -- the dynamic correlation that makes term energies come
out right -- has to be added afterwards, and Kuiva does that with strongly contracted
NEVPT2 on the converged two-component reference.

``NEVPT2`` is post-processing in the strict sense: it consumes converged orbitals, CI
vectors and integral factors, corrects each state's energy, and changes no wavefunction.

The oxygen atom is the cheapest system with something for it to do. 1s2 2s2 2p4 gives a
CAS(4, 6 spinors) whose fifteen determinants are exactly the fifteen states of the 2p^4
configuration: the 3P term (J = 2, 1, 0), the 1D term and the 1S term. Their *fine*
structure comes from spin-orbit coupling and is already right at the CASSCF level; their
*term* separations are dominated by dynamic correlation and are not.

WHAT TO LOOK FOR IN THE OUTPUT
------------------------------
* the correction moving the 1D and 1S terms down towards experiment by thousands of
  wavenumbers while the 3P fine structure barely moves -- two different physics, and the
  right one being corrected;
* the eight excitation classes, printed whether or not they contribute. They are a
  partition of the first-order interacting space, so a sum that leaves one out is not a
  smaller NEVPT2, it is a different quantity -- which is why restricting them (the last
  section) is reported as *partial* and warns;
* the degenerate manifolds surviving the correction. Inside a degenerate manifold the CI
  returns an arbitrary mixture of the members, so the individual per-state corrections
  depend on that arbitrary basis and the barycentre does not. Kuiva therefore reports the
  barycentre *beside* the per-state values with the member spread visible, rather than
  quietly replacing one with the other;
* a frozen core stated as an orbital energy rather than as a count of orbitals -- because a
  count can cut a Kramers pair in half, and an energy cannot.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np

import kuiva
from kuiva.props.multiplet import HARTREE_TO_CM
from kuiva.util import output as out
from kuiva.util import timing
from kuiva.util.logging import add_file_handler, get_logger

NAME = "nevpt2"
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"

log = get_logger("examples." + NAME)

#: The 2p shell of the oxygen atom, and every root of it: C(6, 4) = 15 determinants and 15
#: states, so the state average is complete by construction.
N_ACTIVE, N_ACTIVE_ELEC, N_STATES = 6, 4, 15

#: Freeze the 1s spinors. ⚠ Stated as an orbital ENERGY [Eh], never as a number of
#: orbitals: a count can fall inside a degenerate group and cut a Kramers pair in half,
#: while a threshold on the pseudo-canonical orbital energies cannot. Oxygen's 1s sits near
#: -20.7 Eh and the 2s near -1.3, so -10 separates them with room to spare.
FROZEN_CORE_EH = -10.0

#: O I term energies [cm^-1], NIST Atomic Spectra Database. Anchors, not targets: this is a
#: double-zeta basis, and a term energy is a basis-set-limit quantity.
EXPERIMENT_CM = {"3P1": 158.265, "3P0": 226.977, "1D2": 15867.862, "1S0": 33792.583}

#: A term of a free atom is degenerate as a matter of physics. 0.1 cm^-1 would already be a
#: different physical answer, so the correction may not spread a manifold by more than that.
DEGENERACY_CM = 0.1


def prepare_output() -> Path:
    """Write this run to output/<name>.out, with a scratch spin-orbit cache of its own."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cache = OUTPUT / "amf-cache"
    shutil.rmtree(cache, ignore_errors=True)
    os.environ["KUIVA_AMF_CACHE"] = str(cache)
    path = OUTPUT / (NAME + ".out")
    add_file_handler(path)
    return path


def main() -> int:
    outfile = prepare_output()
    out.banner(log, kuiva.__version__, "example 5: SC-NEVPT2 on the oxygen atom")
    out.entry(log, "output file", str(outfile.relative_to(HERE)))

    # ----------------------------------------------------------------------------------
    # 1. The reference: a state-averaged two-component CASSCF over the whole 2p^4 space.
    # ----------------------------------------------------------------------------------
    oxygen = kuiva.Molecule(atoms=[("O", (0.0, 0.0, 0.0))], basis="x2c-SVPall-2c",
                            charge=0, spin=2)
    scf = kuiva.ScalarSCF(oxygen, memory_gb=2.0).run()
    reference = kuiva.Reference(scf).run()

    # mode="second-order" rather than the default: fifteen equally weighted roots make the
    # orbital problem harder than the CI, and it is the orbital problem that decides the
    # mode. A caller decision, deliberately not inferred from the CI cost.
    # conv_grad below the default: the E2 denominators amplify whatever asymmetry the
    # converged orbitals still carry across a degenerate term (about fifty-fold here), so
    # the landing quality of the reference -- not the correction -- is what decides whether
    # the 0.1 cm^-1 physics check below can pass. An optimizer stopping at |g| = 1e-4 is
    # converged for an energy; it is not necessarily converged for a degeneracy a
    # perturbation is about to magnify.
    cas = kuiva.CASSCF(reference, character=("O", "p"), n_active=N_ACTIVE,
                       n_active_elec=N_ACTIVE_ELEC, n_states=N_STATES,
                       mode="second-order", max_iter=60, conv_grad=1e-6).run()

    out.section(log, "The reference")
    log.info("%s", cas.summary())
    out.entry(log, "state average",
              "complete" if cas.boundary.gap_cm is None
              else "{:.1f} cm^-1 to the next root".format(cas.boundary.gap_cm),
              "", "all {} determinants of the 2p shell".format(N_STATES))

    # ----------------------------------------------------------------------------------
    # 2. The correction.
    # ----------------------------------------------------------------------------------
    # The zeroth-order Hamiltonian is Dyall's, built on the state-averaged spinor Fock
    # operator from the block-equalized density -- the same density the state-averaging gate
    # produced. Level shifts against intruder states are available and are parameter-free by
    # default; any shift that is applied warns, because a shifted energy is a different
    # quantity from an unshifted one.
    pt = kuiva.NEVPT2(cas).run()

    out.section(log, "The finished NEVPT2 stage")
    log.info("%s", pt.summary())

    # ----------------------------------------------------------------------------------
    # 3. The eight excitation classes.
    # ----------------------------------------------------------------------------------
    out.subsection(log, "Excitation classes, state 0")
    table = out.Table(log, [
        out.Column("class", "{}", 12, align="<"),
        out.Column("E2 [Eh]", "{:.9f}", 15),
        out.Column("share [%]", "{:.2f}", 11),
    ])
    table.start()
    total = float(pt.e2[0])
    for name in sorted(pt.class_energies):
        value = float(np.asarray(pt.class_energies[name]).ravel()[0])
        table.row(name, value, 100.0 * value / total if total else 0.0)
    table.end("all eight are registered whether or not they contribute; they partition "
              "the first-order interacting space")
    out.entry(log, "sum over the eight classes", total, "Eh", "", out.E_FMT)
    out.entry(log, "complete (nothing left out)", pt.result.complete)

    # ----------------------------------------------------------------------------------
    # 4. What the correction did to the spectrum.
    # ----------------------------------------------------------------------------------
    # The multiplets of the REFERENCE spectrum, carried unchanged onto the corrected one.
    # Re-grouping the corrected energies would let a manifold that the perturbation split
    # become two manifolds, and the diagnostic would report zero spread for exactly the
    # split it exists to find.
    out.section(log, "Term energies")
    manifolds = pt.multiplets()
    labels = ["3P2", "3P1", "3P0", "1D2", "1S0"][:len(manifolds)]

    table = out.Table(log, [
        out.Column("term", "{}", 8, align="<"),
        out.col_count("states", 8),
        out.Column("CASSCF [cm^-1]", out.CM_FMT, 16),
        out.Column("+NEVPT2 [cm^-1]", out.CM_FMT, 17),
        out.Column("experiment", "{}", 12),
        out.Column("spread [cm^-1]", out.SCI_FMT, 15),
    ])
    table.start()
    e_ref0 = manifolds[0].e_reference
    e_cor0 = manifolds[0].e_corrected
    for label, manifold in zip(labels, manifolds):
        experiment = EXPERIMENT_CM.get(label)
        table.row(label, manifold.size,
                  (manifold.e_reference - e_ref0) * HARTREE_TO_CM,
                  (manifold.e_corrected - e_cor0) * HARTREE_TO_CM,
                  "-" if experiment is None else "{:.1f}".format(experiment),
                  manifold.corrected_spread_cm)
    table.end("the spread column is what the correction did to a manifold that must stay "
              "degenerate")
    out.note(log, "the fine structure (3P2/3P1/3P0) is spin-orbit physics and is already")
    out.note(log, "right without the correction; the TERM separations are correlation and")
    out.note(log, "move by thousands of wavenumbers. The residual against experiment is")
    out.note(log, "basis-set incompleteness -- a term energy is a basis-limit quantity.")

    out.entries(log, [
        ("E2, ground state", float(pt.e2[0]), "Eh", "", out.E_FMT),
        ("E(CASSCF) + E2, ground state", float(pt.total_energies[0]), "Eh", "",
         out.E_FMT),
        ("largest spread inside a manifold",
         max(m.corrected_spread_cm for m in manifolds), "cm^-1", "", out.SCI_FMT),
    ])

    # ----------------------------------------------------------------------------------
    # 5. A frozen core, and what a partial class list does.
    # ----------------------------------------------------------------------------------
    out.section(log, "Options")
    frozen = kuiva.NEVPT2(cas, frozen_core=FROZEN_CORE_EH, report=False).run()
    out.entries(log, [
        ("frozen-core threshold", FROZEN_CORE_EH, "Eh",
         "an orbital energy, never a count of orbitals", "{:.1f}"),
        ("spinors frozen", frozen.result.n_frozen),
        ("E2 with the 1s frozen", float(frozen.e2[0]), "Eh", "", out.E_FMT),
        ("change from the all-electron correction",
         float(frozen.e2[0] - pt.e2[0]), "Eh", "", out.SCI_FMT),
    ])
    out.note(log, "the measured default for both frozen core and deleted virtuals is OFF.")
    out.note(log, "Freezing is a cost decision; state it as an energy so that a threshold")
    out.note(log, "can never fall inside a degenerate group and split a Kramers pair.")

    # Restricting the class list is legal and is reported as what it is. The warning above
    # this line in the output is the point: E2 is then a partial sum, not a cheaper NEVPT2.
    partial = kuiva.NEVPT2(cas, classes=["Sijrs"], report=False).run()
    out.entries(log, [
        ("E2 from the Sijrs class alone", float(partial.e2[0]), "Eh", "", out.E_FMT),
        ("reported complete", partial.result.complete, "",
         "a restricted class list is a PARTIAL sum and says so"),
    ])

    # ----------------------------------------------------------------------------------
    # 6. Assert.
    # ----------------------------------------------------------------------------------
    spread = max(m.corrected_spread_cm for m in manifolds)
    terms_cm = {label: (m.e_corrected - e_cor0) * HARTREE_TO_CM
                for label, m in zip(labels, manifolds)}
    ref_cm = {label: (m.e_reference - e_ref0) * HARTREE_TO_CM
              for label, m in zip(labels, manifolds)}

    checks = {
        "the CASSCF converged": bool(cas.converged),
        "the state average uses the whole determinant space": cas.boundary.gap_cm is None,
        "the 2p^4 configuration gives 3P + 1D + 1S": [m.size for m in manifolds] == [5, 3,
                                                                                     1, 5,
                                                                                     1],
        "the correction is complete (all eight classes)": bool(pt.result.complete),
        "the correction lowers every state": bool(np.all(np.asarray(pt.e2) < 0.0)),
        "every manifold stays degenerate to {} cm^-1".format(DEGENERACY_CM):
            spread < DEGENERACY_CM,
        "1D moves towards experiment":
            abs(terms_cm["1D2"] - EXPERIMENT_CM["1D2"]) < abs(ref_cm["1D2"]
                                                              - EXPERIMENT_CM["1D2"]),
        "1S moves towards experiment":
            abs(terms_cm["1S0"] - EXPERIMENT_CM["1S0"]) < abs(ref_cm["1S0"]
                                                              - EXPERIMENT_CM["1S0"]),
        "freezing the 1s freezes exactly the two 1s spinors": frozen.result.n_frozen == 2,
        "a restricted class list is reported as partial": not partial.result.complete,
    }
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
