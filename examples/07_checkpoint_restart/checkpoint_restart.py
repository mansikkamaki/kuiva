"""Example 7 -- checkpointing: a CASSCF interrupted part way through and resumed.

    source setup.sh          # once per shell
    python checkpoint_restart.py

Runs in three to four minutes on TiCl3 and writes ``output/checkpoint_restart.out``.

WHAT THIS SHOWS
---------------
A production CASSCF on a heavy-element complex runs for hours, and jobs get killed. Kuiva
checkpoints every macro-iteration into a schema-versioned HDF5 file, and a restart
*continues* the trajectory rather than beginning a new one.

The example runs the same calculation twice: once straight through, and once stopped at
iteration 6 and resumed from disk. The two must land on the same energy, in a comparable
total number of macro-iterations -- ``max_iter`` counts *total* iterations across the
restart, so an interrupted run costs what an uninterrupted one would.

WHAT IS IN THE FILE, AND WHAT DELIBERATELY IS NOT
-------------------------------------------------
Always written, because they are small and precious: the orbitals and the orbital-rotation
state, the active-space density matrices, the converged state energies, and the metadata
that identifies the calculation.

Never written, because they regenerate deterministically from the orbitals: the four-index
integrals and the density-fitting or Cholesky intermediates. A restart rebuilds them. For a
heavy element the expensive part of that rebuild would be the four-component atomic
calculation behind the two-electron spin-orbit term -- which is why *that* has a persistent
cache of its own, keyed on the element rather than on the iterate.

Written by policy under a budget: the CI vectors, and only below a size threshold. Kuiva
estimates the write cost and *thins* -- drops the vectors -- or skips outright when a
checkpoint would cost more than a few per cent of the compute since the last one, but a
checkpoint at a *converged* iteration is always written. The vectors are a Davidson warm
start, which is why they are the first thing dropped: losing them costs the next solve its
order of magnitude, and nothing else. A thinned checkpoint still restarts the calculation.

A ``solver="dmrg"`` CASSCF checkpoints too, into **two** files: this same trajectory file
(without CI vectors -- the network has none) and a sibling ``*.network.h5`` holding the
network state itself, written rolling at the end of each completed sweep. A restart resumes
the trajectory exactly and warm-starts the network from the sibling; example 4 shows the
two files. Everything below is the conventional-CI route.

⚠ **Same energy, not the same bits.** The checkpoint restores the orbitals, the quasi-Newton
curvature memory, the trust radius and the eigensolver's guess exactly. What it cannot
restore is that an iterative eigensolver is deterministic only to its residual tolerance, so
the comparison below is made at that tolerance -- far below the optimizer's own convergence
threshold, which is what keeps it meaningful.

Failure semantics are the opposite of a cache's, and deliberately so: a failed *write* is a
warning and the run continues; a failed *read* on an explicitly requested restart is an
error that propagates, because silently starting over wastes exactly the hours the file
existed to protect. A schema-version mismatch refuses outright.

STOPPING BEFORE THE ALLOCATION DOES
-----------------------------------
The other half of surviving a batch queue. `deadline=` makes the run stop *itself* while
there is still time to write a checkpoint, instead of being killed at the wall with only
the last one to show for it. Three things about it are worth knowing:

* ⚠ **there is no default.** A cluster with no queue time limit is an ordinary place to
  run, and a deadline invented for such a run could only ever end it early for no reason.
  `deadline=None` reads nothing, prints nothing and stops nothing;
* the two ways to set one: a budget of your own (`deadline=300`, `"6h"`, `"24:00:00"`), or
  the batch allocation's own limit -- `deadline="slurm"`, read once from
  `$SLURM_JOB_END_TIME` and then from `scontrol`, which **refuses** if it cannot be read
  because a request that silently produced no deadline is the worst of the outcomes.
  `deadline="auto"` is the portable spelling: that limit where there is one, none where
  there is not, stated either way;
* ⚠ **the decision is predictive.** Stopping when the time is already spent is stopping too
  late, because the checkpoint is written *afterwards*. The run stops when the time left is
  less than the longest recent macro-iteration plus the estimated write plus a stated
  margin -- and the final write is then forced past the cadence rules, because that
  checkpoint is the result rather than insurance.

`signals=` is the other half: the kill nobody announced -- `scancel`, a preemption, a
`SIGTERM` at the wall. `signals=True` catches SIGTERM/SIGUSR1/SIGUSR2 and stops the run at
the next macro-iteration boundary with its checkpoint written, through the same decision the
deadline uses. This example arms it below, where nothing will send one; a batch script buys
the lead time with `#SBATCH --signal=B:USR1@600`.

⚠ Three things about it are deliberate, and the first is why it is not simply always on:
handlers are **opt-in and installed only while the stage runs**, because a library that
installs them behind your back breaks embedding, test runners and notebooks; a **second**
signal is not waited for, so a repeated kill acts at once; and the request is **process-wide
and outlives the stage**, so the next long stage refuses to start rather than beginning work
this process will not live to finish. That last one is what makes a signalled run *exit*
rather than merely pause.

WHAT TO LOOK FOR IN THE OUTPUT
------------------------------
* the interrupted run stopping cleanly from inside the optimizer's callback -- a run that
  stops itself keeps everything it finished; one killed by an external timeout keeps
  nothing;
* the checkpoint's contents, including the identity of the CI space, which is what makes
  curvature memory chart-scoped: resuming into a *different* CI space clears the curvature
  rather than transporting it onto a surface it does not belong to;
* the resumed run continuing the iteration count rather than restarting it, and landing on
  the same energy;
* the active space coming *from the file* on restart. It is not restated here; a restart
  continues the calculation that was interrupted, and a restated space that disagreed with
  the file would be refused rather than reconciled.
"""
from __future__ import annotations

import math
import os
import shutil
import signal
from pathlib import Path
from typing import List

import kuiva
from kuiva.io.checkpoint import read_checkpoint
from kuiva.util import output as out
from kuiva.util import timing
from kuiva.util.logging import add_file_handler, get_logger

NAME = "checkpoint_restart"
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"

log = get_logger("examples." + NAME)

#: Planar D3h TiCl3, Ti-Cl [Angstrom] -- the calculation of example 3.
R_TICL = 2.25

#: The 3d shell, the ground Kramers doublet.
N_ACTIVE, N_ACTIVE_ELEC, N_STATES = 10, 1, 2

#: Budgets, all explicit. The wall budget is a `deadline=`, enforced from inside the
#: optimizer so that termination never depends on convergence, and a stopped run keeps what
#: it finished. In seconds, as a number: a bare numeric *string* is refused, because "300"
#: is five hours to Slurm's --time and would be five minutes here.
MAX_ITER, CONV_GRAD, INTERRUPT_AT, WALL_BUDGET_S = 40, 1.0e-4, 6, 300.0

#: The restarted and the uninterrupted run are compared at the eigensolver's own residual
#: tolerance, not tighter. See the module docstring.
RESTART_ENERGY_TOL = 1.0e-7


def planar_mx3(metal: str, ligand: str, r: float) -> List[tuple]:
    """Planar D3h MX3, metal at the origin, ligands in the xy plane."""
    atoms = [(metal, (0.0, 0.0, 0.0))]
    for k in range(3):
        theta = 2.0 * math.pi * k / 3.0
        atoms.append((ligand, (r * math.cos(theta), r * math.sin(theta), 0.0)))
    return atoms


def prepare_output() -> Path:
    """Write this run to output/<name>.out, with a scratch spin-orbit cache of its own."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cache = OUTPUT / "amf-cache"
    shutil.rmtree(cache, ignore_errors=True)
    os.environ["KUIVA_AMF_CACHE"] = str(cache)
    path = OUTPUT / (NAME + ".out")
    add_file_handler(path)
    return path


def stop_after(iteration: int):
    """Simulate an interrupted job: stop cleanly at ``iteration``, keeping the checkpoint.

    The optimizer's callback sees one dict per macro-iteration and returns False to stop.
    This one is a demonstration; the wall-clock budget beside it is `deadline=` below, and
    the two coexist on the single hook because the checkpoint policy chains them.
    """
    def callback(info: dict):
        if info["iteration"] >= iteration:
            log.warning("stopping at iteration %d to demonstrate the restart; the "
                        "checkpoint written at this iteration is what the second leg "
                        "resumes from", info["iteration"])
            return False
        return None
    return callback


def main() -> int:
    outfile = prepare_output()
    out.banner(log, kuiva.__version__, "example 7: checkpoint and restart")
    out.entry(log, "output file", str(outfile.relative_to(HERE)))

    # ----------------------------------------------------------------------------------
    # 1. The front end, once. Both CASSCFs below start from identical orbitals, which is
    #    the only way the comparison means anything.
    # ----------------------------------------------------------------------------------
    molecule = kuiva.Molecule(atoms=planar_mx3("Ti", "Cl", R_TICL),
                              basis="x2c-SVPall-2c", charge=0, spin=1)
    scf = kuiva.ScalarSCF(molecule, memory_gb=6.0).run()
    reference = kuiva.Reference(scf).run()

    straight_chk = OUTPUT / "ticl3_straight.chk"
    interrupted_chk = OUTPUT / "ticl3_interrupted.chk"
    # Cleared first. An example whose result depends on what is already on the disk is not
    # a demonstration of anything -- the same rule as the scratch spin-orbit cache above.
    for stale in (straight_chk, interrupted_chk):
        if stale.exists():
            stale.unlink()

    selection = dict(character=("Ti", "d"), n_active=N_ACTIVE,
                     n_active_elec=N_ACTIVE_ELEC)

    out.section(log, "Problem")
    out.entries(log, [
        ("system", "TiCl3", "", "planar D3h, Ti-Cl = {:.2f} A".format(R_TICL)),
        ("active space", "CAS({}, {})".format(N_ACTIVE_ELEC, N_ACTIVE), "",
         "the Ti 3d shell, by orbital character"),
        ("states", N_STATES, "", "the ground Kramers doublet"),
        ("macro-iteration budget", MAX_ITER, "", "total, across any restart"),
        ("interrupt after", INTERRUPT_AT, "iterations"),
    ])

    # ----------------------------------------------------------------------------------
    # 2. The uninterrupted run, checkpointing every macro-iteration.
    # ----------------------------------------------------------------------------------
    out.section(log, "CASSCF, uninterrupted")
    # deadline=<seconds> is a budget of this run's own. In a batch job it is instead
    # deadline="slurm" -- the allocation's own limit, read from the scheduler -- or
    # deadline="auto", which uses that limit where there is one and runs without a deadline
    # where there is not. ⚠ There is no default: a cluster with no time limit is an ordinary
    # place to run, and an invented deadline could only ever end such a run early.
    with timing.timer("CASSCF (uninterrupted)") as t_full:
        full = kuiva.CASSCF(reference, n_states=N_STATES, max_iter=MAX_ITER,
                            conv_grad=CONV_GRAD, checkpoint=straight_chk,
                            deadline=WALL_BUDGET_S, signals=True, **selection).run()
    log.info("%s", full.summary())

    # ----------------------------------------------------------------------------------
    # 3. The same calculation, interrupted.
    # ----------------------------------------------------------------------------------
    out.section(log, "CASSCF, interrupted at iteration {}".format(INTERRUPT_AT))
    with timing.timer("CASSCF (first leg)") as t_leg1:
        first_leg = kuiva.CASSCF(reference, n_states=N_STATES, max_iter=MAX_ITER,
                                 conv_grad=CONV_GRAD, checkpoint=interrupted_chk,
                                 callback=stop_after(INTERRUPT_AT),
                                 deadline=WALL_BUDGET_S, signals=True,
                                 **selection).run()

    stored = read_checkpoint(interrupted_chk)
    out.subsection(log, "What is on disk")
    stored.report(log)
    out.entries(log, [
        ("checkpoint file", str(interrupted_chk.relative_to(HERE))),
        ("size", interrupted_chk.stat().st_size / 1024.0, "kB", "", "{:.1f}"),
        ("CI space identity", stored.space_key, "", "what makes curvature chart-scoped"),
        ("Hamiltonian provenance stored", "hamiltonian" in stored.metadata),
    ])

    # ----------------------------------------------------------------------------------
    # 4. Resume.
    # ----------------------------------------------------------------------------------
    # ⚠ The active space is NOT restated: it comes from the file. Restating it in a way that
    # disagreed with what is stored would be refused rather than reconciled, because a
    # restart continues the calculation that was interrupted.
    out.section(log, "CASSCF, resumed from the checkpoint")
    with timing.timer("CASSCF (resumed)") as t_leg2:
        resumed = kuiva.CASSCF(reference, n_states=N_STATES, max_iter=MAX_ITER,
                               conv_grad=CONV_GRAD, restart=interrupted_chk,
                               deadline=WALL_BUDGET_S, signals=True).run()
    log.info("%s", resumed.summary())

    out.entries(log, [
        ("iterations before the interruption", first_leg.orbital.n_iterations),
        ("iterations after the restart",
         resumed.orbital.n_iterations - first_leg.orbital.n_iterations),
        ("total", resumed.orbital.n_iterations, "",
         "against {} uninterrupted".format(full.orbital.n_iterations)),
        ("difference from the uninterrupted energy",
         abs(resumed.energy - full.energy), "Eh", "", out.SCI_FMT),
    ])

    # ----------------------------------------------------------------------------------
    # 5. Side by side.
    # ----------------------------------------------------------------------------------
    out.section(log, "Summary")
    table = out.Table(log, [
        out.Column("run", "{}", 30, align="<"),
        out.col_iter("iter"),
        out.col_energy("E [Eh]"),
        out.col_resid("|grad|"),
        out.Column("conv", "{}", 6),
        out.col_time("wall [s]"),
    ])
    table.start()
    table.row("uninterrupted", full.orbital.n_iterations, full.energy,
              full.orbital.grad_norm, "yes" if full.converged else "NO", t_full.wall)
    table.row("interrupted + resumed", resumed.orbital.n_iterations, resumed.energy,
              resumed.orbital.grad_norm, "yes" if resumed.converged else "NO",
              t_leg1.wall + t_leg2.wall)
    table.end("wall times differ between the two rows by whatever the machine was doing; "
              "read cpu [s] in the timing table for cost")

    # ----------------------------------------------------------------------------------
    # 6. Assert.
    # ----------------------------------------------------------------------------------
    checks = {
        "the uninterrupted run converged": bool(full.converged),
        "the resumed run converged": bool(resumed.converged),
        "the restart reproduces the energy to {:.0e} Eh".format(RESTART_ENERGY_TOL):
            abs(resumed.energy - full.energy) < RESTART_ENERGY_TOL,
        "the restart continued the iteration count":
            resumed.orbital.n_iterations > first_leg.orbital.n_iterations,
        "the interruption happened where it was asked to":
            first_leg.orbital.n_iterations == INTERRUPT_AT,
        "the checkpoint carries the active space":
            (stored.n_active_elec == N_ACTIVE_ELEC
             and stored.spaces.n_active == N_ACTIVE),
        "the checkpoint carries the CI space identity":
            stored.space_key == "full-ci:{}:{}".format(N_ACTIVE, N_ACTIVE_ELEC),
        "the converged checkpoint of the straight run exists": straight_chk.exists(),
        # ⚠ The deadline is a safety net here, not the mechanism under test: these runs
        # converge well inside it. A fired deadline would mean the box was far slower than
        # the budget assumed, and every number above would then be an iterate rather than
        # an answer -- which is exactly what the check is for.
        "the wall budget was never needed": not any(
            stage.deadline.fired for stage in (full, first_leg, resumed)),
        # ⚠ Nothing sends a signal here, so this asserts the quiet half of the contract: an
        # armed handler that was never used changes nothing at all, and gives the process's
        # own handlers back when the stage ends.
        "no signal arrived, and none of the arming changed anything": (
            not any(stage.signals.fired for stage in (full, first_leg, resumed))
            and signal.getsignal(signal.SIGTERM) is signal.SIG_DFL),
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
