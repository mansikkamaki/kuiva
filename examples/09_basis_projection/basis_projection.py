"""Example 9 -- converge the CASSCF in a small basis, then continue it in a large one.

    source setup.sh          # once per shell
    python basis_projection.py

Runs in about four minutes on N2 and writes ``output/basis_projection.out``.

WHAT THIS SHOWS
---------------
A CASSCF costs what its basis costs, and almost all of that expense buys the *orbitals*.
The active space -- which orbitals are correlated, and what they look like -- is decided by
chemistry and is very nearly the same in a small basis as in a large one. So the standard
production route is to converge the calculation where it is cheap and carry the result into
the basis you actually want:

    ScalarSCF(small) -> Reference -> CASSCF          the cheap calculation
    ScalarSCF(large) -> Reference -> CASSCF(project_from=the cheap one)

The middle step is a basis-set projection: the converged spinors are re-expressed over the
larger AO basis, made orthonormal again, and the extra dimensions the larger basis brings are
completed from its own SCF. It is one keyword.

On the N2 CAS(8, 12) below the projected run reaches the converged answer in about 7
macro-iterations where the direct route needs about 14 -- both to the same energy, to better
than 1e-8 Eh. Half the iterations, in the basis where an iteration is expensive.

⚠ Iteration counts move by one either way between runs of identical arithmetic: the
trajectory is a sequence of accept/reject decisions on a quadratic model, and a threaded
reduction reorders at the last bit. Only the ratio is the point, and what this example asserts
is what cannot drift -- never the counts themselves.

EIGHT THINGS WORTH WATCHING
---------------------------
1. **The active space holds the SCF reference determinant, and that is not automatic.** It is
   the N2 valence space -- the p-derived orbitals of both nitrogens, four occupied pairs and
   two empty ones -- selected by AVAS, which projects the orbitals onto the free-atom N 2p
   reference and keeps the combinations carrying that character. It is stated as a projection
   with a threshold, never as an index window, and it selects the **same six pairs in both
   bases**. The example checks the property that matters and cannot be assumed: every orbital
   the SCF occupies lies inside inactive + active, so the CAS can represent the reference
   determinant. A space that cannot is not merely a poorer one -- it is where the symmetric
   solution stops being a minimum (point 7).

2. **The active space is not restated on the projected run, and may not be.** It was chosen
   once, against the orbitals being carried, and it comes across with them. Re-selecting it
   here -- by character, by AVAS or by index -- would resolve it against *this* reference's
   guess orbitals instead, which is a different calculation wearing the same name. Kuiva
   refuses that rather than doing it.

3. **What crosses the basis change is the active space, not the whole orbital set.** That is
   `carry="active"`, the default, and it deliberately carries *less* than it could. The
   inactive orbitals of the small-basis calculation are not eigenvectors of anything in the
   large basis: carrying them re-introduces an inactive-virtual gradient -- and the core
   orbital energies are the largest numbers in that block -- which the large basis' own SCF
   had already removed. Measured across three systems and both directions, `carry="all"` costs
   between nothing and twice the macro-iterations, and never fewer. The run below prints both,
   so the difference is visible rather than asserted.

4. **The projection is measured, not assumed.** Three numbers say whether it is worth using,
   and all three are invariants rather than coefficient comparisons:

   * the **retained norm** of each source orbital, which is how much of it exists in the
     target basis at all;
   * the **principal overlaps** between the source active space and the space handed on --
     cosines of the principal angles, unchanged by any rotation inside either space, so they
     are a statement about the space and not about the arbitrary basis an eigensolver
     returned;
   * the **complement separation**, which is the evidence that the orbitals with no source
     really are the orthogonal complement and not a numerically ambiguous slice of it.

   None of these is decoration. Every way of getting a basis projection wrong still produces
   an orthonormal orbital set of the right shape that starts a calculation which converges;
   the failure shows up only as a run that takes longer, which is exactly what nobody
   investigates.

5. **A guess may change the cost and may not change the answer.** Both routes reach the same
   CASSCF solution -- asserted here as an energy agreement *and* as the variational statement
   that the projected route is not above the direct one.

6. **The orbitals handed over are Kramers pairs, and the source's need not be.** A converged
   CASSCF is entitled to leave its active orbitals as an arbitrary mixture of the pairs --
   rotations inside the active space are redundant, so nothing in the optimization pushes
   back. The projection rebuilds the pairs on the carried block before it builds anything
   against it, because every consumer of the convention downstream needs them. It is free: it
   is a rotation inside a space the optimizer had already left arbitrary.

7. **The time-reversal stability line is a statement about the answer, not the run.** The
   orbital optimizer holds the spinors Kramers paired while it works, which is a constraint,
   so at the end it measures the curvature of the orbital Hessian along the rotations that
   constraint forbids. Non-negative -- as it is here, on a valence space that holds the
   reference determinant -- means the time-reversal-symmetric solution really is a minimum and
   the run is finished. Negative would mean it is a saddle, and Kuiva would release the
   constraint, step off it and carry on; that is what the *old* active space of this example
   did, before it was replaced.

8. **The reverse direction works and is not the inverse.** The last section carries the
   converged large-basis result *back down* to the small basis. Going down throws away a
   variational space rather than reproducing one, so orbitals lose norm and the diagnostics
   stop being near 1 -- which is the honest report, and the run states it.

THE ANSWER YOU SHOULD SEE
-------------------------
N2 at 1.098 A, CAS(8, 12) over the p-derived valence orbitals:

    x2c-SVPall-2c    E = -109.012788  (the cheap calculation)
    x2c-TZVPall-2c   E = -109.148098  from either route, to better than 1e-8 Eh

and the projected route reaching the second number in about half the macro-iterations the
direct one needs. The absolute energies are a valence CASSCF without dynamic correlation and
are not spectroscopy; what this example is about is the route, not the number.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import kuiva
from kuiva.spinor.expand import time_reverse
from kuiva.util import output as out
from kuiva.util import resources as res
from kuiva.util import timing
from kuiva.util.logging import add_file_handler, get_logger

NAME = "basis_projection"
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"

log = get_logger("examples." + NAME)

#: N2 bond length [Angstrom].
R_NN = 1.098

#: The small basis the calculation is converged in, and the one it is continued in.
SMALL_BASIS, LARGE_BASIS = "x2c-SVPall-2c", "x2c-TZVPall-2c"

#: The valence active space, as a projection onto the free-atom N 2p orbitals pooled over
#: both nitrogens: six Kramers pairs, four of them occupied, holding eight electrons --
#: CAS(8, 12 spinors). ⚠ Stated this way rather than as `character=`, because a *character*
#: selection takes the lowest orbitals that carry the character, and in these two bases that
#: is not the same set: the TZVP core pairs carry more than half their Loewdin population on
#: p functions. AVAS projects instead, and selects the same six pairs in both.
AVAS = dict(atom=[0, 1], l="p", threshold=0.5)

#: Budgets, explicit, so that termination never depends on convergence. Both routes are run
#: to convergence -- the comparison is of iteration counts, so both have to reach the same
#: place first.
MAX_ITER, CONV_GRAD = 40, 1.0e-4

#: The cheap run gets room to actually converge -- it is the calculation everything else is
#: built on, and at ~0.5 s per macro-iteration in this basis the budget costs nothing. ⚠ It is
#: generous on purpose: a run that stops one iteration short of its budget is a knife-edge an
#: example may not assert on, and the pure-NumPy path needs more iterations here than the
#: compiled one does.
CHEAP_MAX_ITER = 80

#: mode="second-order" is the right explicit choice here -- it is the orbital problem that
#: decides the mode, not the CI, and this one is a genuine multireference orbital problem.
MODE = "second-order"

#: A projected active space that overlapped the one it came from by less than this would not
#: be the same calculation any more. Measured here at 0.9999.
FIDELITY_MIN = 0.99


def prepare_output() -> Path:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / (NAME + ".out")
    add_file_handler(path)
    return path


def nitrogen(basis: str) -> kuiva.Molecule:
    return kuiva.Molecule(atoms=[("N", (0.0, 0.0, 0.0)), ("N", (0.0, 0.0, R_NN))],
                          basis=basis)


def scalar_scf(basis: str) -> kuiva.ScalarSCF:
    """⚠ ``atomic_reference=True`` because AVAS projects onto the free-atom orbitals, and
    those are built at ingestion -- the analysis layer has no integral library and cannot
    make them later."""
    return kuiva.ScalarSCF(nitrogen(basis), memory_gb=6.0, screening="none",
                           atomic_reference=True).run()


def kramers_deviation(coeff: np.ndarray) -> float:
    """``max |c_2p+1 - T c_2p|`` -- the pairing convention as an identity on coefficients.

    It needs no metric, which is the point: it is a statement about the columns and holds in
    the AO basis exactly as it does in the orthonormal one. A projection that treated the two
    spin blocks differently would break it, and nothing else in this example would notice.
    """
    return float(np.abs(coeff[:, 1::2] - time_reverse(coeff[:, ::2])).max())


def holds_reference_determinant(scf: kuiva.ScalarSCF, cas) -> bool:
    """Is every SCF-occupied orbital inside inactive + active?

    The property the whole example rests on, and the one a poorly chosen active space fails
    silently. It is checked on *spatial* orbital indices lifted to spinors (`p` -> `2p`,
    `2p+1`), which is legitimate here because AVAS rotates only **within** groups of equal
    occupation: the occupied span it hands over is the SCF's own.
    """
    occupied = np.nonzero(np.asarray(scf.data.mo_occ) > 0.0)[0]
    spinors = np.concatenate([2 * occupied, 2 * occupied + 1])
    spaces = cas.active.spaces
    inside = set(spaces.inactive.tolist()) | set(spaces.active.tolist())
    return set(spinors.tolist()) <= inside


def projection_report(projection) -> None:
    """The three invariants that say whether a projection is worth using."""
    active = projection.overlaps["active"]
    out.entries(log, [
        ("orbitals carried across", projection.n_carried, "",
         "carry={}".format(projection.carry)),
        ("orbitals taken from the target's own SCF",
         projection.plan.n_target - projection.n_carried),
        ("retained norm of the carried orbitals (min)",
         float(projection.retained[projection.plan.active].min()), "", "", "{:.6f}"),
        ("active-space principal overlaps (min / max)",
         "{:.6f} / {:.6f}".format(float(active.min()), float(active.max()))),
        ("complement separation (kept / discarded)",
         "{:.2e} / {:.2e}".format(*projection.complement_gap)),
    ])


def main() -> int:
    outfile = prepare_output()
    out.banner(log, kuiva.__version__, "example 9: a CASSCF continued in a larger basis")
    out.entry(log, "output file", str(outfile.relative_to(HERE)))

    out.section(log, "Problem")
    out.entries(log, [
        ("system", "N2", "", "r(N-N) = {:.3f} A".format(R_NN)),
        ("active space", "CAS(8, 12)", "",
         "AVAS on the free-atom N 2p, both centres"),
        ("cheap basis", SMALL_BASIS),
        ("production basis", LARGE_BASIS),
        ("orbital optimizer", MODE, "", "max_iter = {}".format(MAX_ITER)),
    ])
    out.note(log, "screening='none': the two-electron spin-orbit picture change is Kuiva's")
    out.note(log, "default and changes no scalar quantity, so on a light-atom convergence")
    out.note(log, "demonstration it is pure cost. Leave it on for anything with a heavy atom.")

    # ----------------------------------------------------------------------------------
    # 1. The cheap calculation. This is the one that decides the orbitals.
    # ----------------------------------------------------------------------------------
    out.section(log, "1. CASSCF in the cheap basis")
    small_scf = scalar_scf(SMALL_BASIS)
    small = kuiva.Reference(small_scf).run()
    cheap = kuiva.CASSCF(small, avas=AVAS, mode=MODE, max_iter=CHEAP_MAX_ITER,
                         conv_grad=CONV_GRAD, report=False).run()
    log.info("%s", cheap.summary())

    # ----------------------------------------------------------------------------------
    # 2. The production basis, twice: from its own SCF guess, and from the projection.
    # ----------------------------------------------------------------------------------
    large_scf = scalar_scf(LARGE_BASIS)
    large = kuiva.Reference(large_scf).run()

    out.section(log, "2. The production basis from its own SCF guess")
    out.note(log, "the same active space, selected again in this basis by the same AVAS")
    out.note(log, "request -- and it is the same six pairs, which is why the comparison is")
    out.note(log, "a comparison of routes rather than of calculations.")
    direct = kuiva.CASSCF(large, avas=AVAS, mode=MODE, max_iter=MAX_ITER,
                          conv_grad=CONV_GRAD, report=False).run()
    log.info("%s", direct.summary())

    out.section(log, "3. The production basis, continued from the cheap calculation")
    # ⚠ No active space is given, and none may be: it comes across with the orbitals. The
    # only new argument is project_from=.
    projected = kuiva.CASSCF(large, project_from=cheap, mode=MODE, max_iter=MAX_ITER,
                             conv_grad=CONV_GRAD).run()
    log.info("%s", projected.summary())

    out.subsection(log, "What the projection carried")
    projection_report(projected.projection)
    # ⚠ On the PROJECTION's own output, not on the converged CASSCF orbitals: the optimizer
    # is entitled to rotate the active pairs into each other afterwards, which is redundant
    # and changes nothing.
    out.entry(log, "Kramers pairing deviation of the projected orbitals",
              kramers_deviation(projected.projection.coeff), fmt="{:.2e}",
              note="pairs rebuilt on the carried block before the complement")

    out.subsection(log, "The two routes to the same number")
    out.entries(log, [
        ("E(CASSCF), from the SCF guess", direct.energy, "Eh", "", out.E_FMT),
        ("E(CASSCF), from the projection", projected.energy, "Eh", "", out.E_FMT),
        ("difference", abs(direct.energy - projected.energy), "Eh", "", out.SCI_FMT),
        ("macro-iterations, from the SCF guess", direct.orbital.n_iterations, "",
         "converged" if direct.converged else "budget exhausted"),
        ("macro-iterations, from the projection", projected.orbital.n_iterations, "",
         "converged" if projected.converged else "budget exhausted"),
        ("|grad|, from the SCF guess", direct.orbital.grad_norm, "", "", out.SCI_FMT),
        ("|grad|, from the projection", projected.orbital.grad_norm, "", "", out.SCI_FMT),
    ])

    out.subsection(log, "Is the converged solution a minimum?")
    out.note(log, "the curvature of the orbital Hessian along the time-reversal-odd")
    out.note(log, "rotations the Kramers constraint forbids, measured at the converged")
    out.note(log, "point. Positive: the symmetric solution is a minimum and the run is done.")
    out.entries(log, [
        ("lowest time-odd curvature, direct", direct.orbital.time_odd_curvature,
         "Eh/rad^2", "", out.SCI_FMT),
        ("lowest time-odd curvature, projected", projected.orbital.time_odd_curvature,
         "Eh/rad^2", "", out.SCI_FMT),
        ("constraint released", projected.orbital.kramers_released, "",
         "a release would mean the symmetric solution is a saddle"),
    ])

    # ----------------------------------------------------------------------------------
    # 3. The other carry policy, so the default is visible rather than asserted.
    # ----------------------------------------------------------------------------------
    out.section(log, "4. Carrying the whole orbital set instead (carry='all')")
    out.note(log, "the non-default policy, shown for contrast: it carries the small basis'")
    out.note(log, "inactive orbitals too, which are not eigenvectors of anything here.")
    carried_all = kuiva.CASSCF(large, project_from=cheap, mode=MODE, max_iter=MAX_ITER,
                               conv_grad=CONV_GRAD, report=False,
                               projection=dict(carry="all")).run()
    out.entries(log, [
        ("macro-iterations, carry='active'", projected.orbital.n_iterations),
        ("macro-iterations, carry='all'", carried_all.orbital.n_iterations, "",
         "converged" if carried_all.converged else "budget exhausted"),
    ])

    # ----------------------------------------------------------------------------------
    # 4. Downwards, which is the same call and not the inverse.
    # ----------------------------------------------------------------------------------
    out.section(log, "5. The reverse direction: production basis back onto the cheap one")
    out.note(log, "the same call. Going down discards a variational space rather than")
    out.note(log, "reproducing one, so the retained norms are no longer near 1 and say so.")
    back = kuiva.CASSCF(small, project_from=projected, mode=MODE, max_iter=MAX_ITER,
                        conv_grad=CONV_GRAD, report=False).run()
    down = back.projection
    out.entries(log, [
        ("virtual orbitals the small basis cannot hold", down.plan.n_dropped),
        ("retained norm over ALL source orbitals (min)", float(down.retained.min()),
         "", "", "{:.6f}"),
        ("retained norm over the carried active space (min)",
         float(down.retained[down.plan.active].min()), "", "", "{:.6f}"),
        ("active-space principal overlap (min)",
         float(down.overlaps["active"].min()), "", "", "{:.6f}"),
        ("E(CASSCF) in the cheap basis, direct", cheap.energy, "Eh", "", out.E_FMT),
        ("E(CASSCF) in the cheap basis, from above", back.energy, "Eh", "", out.E_FMT),
        ("macro-iterations", back.orbital.n_iterations, "",
         "converged" if back.converged else "budget exhausted"),
    ])

    # ----------------------------------------------------------------------------------
    # 5. What the example claims.
    # ----------------------------------------------------------------------------------
    # ⚠ Iteration COUNTS are not asserted as numbers -- they are a measurement and they move
    # with the machine. What is asserted is the inequality (7 against 14 here), the
    # variational statement, and the invariants of the projection itself.
    checks = {
        "the active space holds the SCF reference determinant":
            holds_reference_determinant(small_scf, cheap)
            and holds_reference_determinant(large_scf, direct),
        "the cheap CASSCF converged": bool(cheap.converged),
        "both large-basis routes converged":
            bool(direct.converged) and bool(projected.converged),
        "the projected route needed fewer macro-iterations":
            projected.orbital.n_iterations < direct.orbital.n_iterations,
        "and reached the same energy, from below":
            projected.energy <= direct.energy + 1e-9
            and abs(projected.energy - direct.energy) < 1e-6,
        "the projected active space is the one it came from":
            projected.projection.fidelity > FIDELITY_MIN,
        "the active space was inherited, not re-selected":
            "projected from" in projected.active.description,
        "the carried active orbitals exist in the target basis":
            float(projected.projection.retained[
                projected.projection.plan.active].min()) > 0.99,
        "the completed orbitals are a clean orthogonal complement":
            projected.projection.complement_gap[0] > 0.5
            and projected.projection.complement_gap[1] < 1e-8,
        "the projection delivers exact Kramers pairs":
            kramers_deviation(projected.projection.coeff) < 1e-10,
        "the time-reversal-symmetric solution is a minimum, by both routes":
            direct.orbital.time_odd_curvature > 0.0
            and projected.orbital.time_odd_curvature > 0.0
            and not projected.orbital.kramers_released,
        "the reverse projection converged to the cheap basis' own solution":
            bool(back.converged) and abs(back.energy - cheap.energy) < 1e-7,
        "and it reports the norm it could not carry":
            float(down.retained.min()) < 0.99,
    }
    failures = report(checks)

    timing.summary(log)
    res.summary(log)
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
