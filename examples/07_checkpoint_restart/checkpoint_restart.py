"""Example 7 -- checkpointing: a CASSCF interrupted, resumed, and finished from its file.

    source setup.sh          # once per shell
    python checkpoint_restart.py

Runs in a few minutes on TiCl3 and writes ``output/checkpoint_restart.out``.

WHAT THIS SHOWS
---------------
A production CASSCF on a heavy-element complex runs for hours, and jobs get killed. Kuiva
checkpoints every macro-iteration into a schema-versioned HDF5 file, and a restart
*continues* the trajectory rather than beginning a new one.

The example runs the same calculation twice: once straight through, and once stopped at
iteration 6 and resumed from disk. The two must land on the same energy, in a comparable
total number of macro-iterations -- ``max_iter`` counts *total* iterations across the
restart, so an interrupted run costs what an uninterrupted one would.

It then does two things a *finished* file makes possible. ``CASSCF.from_checkpoint`` reads
the converged run back as a stage that reports itself converged and feeds everything
downstream, so a dump or a correction that was never requested does not cost the CASSCF
again. And SC-NEVPT2 gets a checkpoint of its own, written per excitation class.

WHAT IS IN THE FILE, AND WHAT DELIBERATELY IS NOT
-------------------------------------------------
Always written, because they are small beside the orbitals or because they say which
calculation the orbitals belong to: the orbitals themselves, the orbital partition, the
1-RDM, the converged state energies, the state average, the Hamiltonian provenance and a
fingerprint of the *system* -- the molecule, the basis, the nuclear model, the gauge origin.
That last one is what makes a restart against the wrong file an error rather than a
plausible number: an orthonormal set of the right dimension optimizes to something whatever
molecule it came from.

Never written, because they regenerate deterministically from the orbitals: the four-index
integrals and the density-fitting or Cholesky intermediates, the 3- and 4-RDMs, and the
transition densities. A restart rebuilds them. For a heavy element the expensive part of
that rebuild would be the four-component atomic calculation behind the two-electron
spin-orbit term -- which is why *that* has a persistent cache of its own, keyed on the
element rather than on the iterate.

Written by policy, and given up in a fixed order when the budget bites. ⚠ **The order is by
what it costs to get the thing back, not by what it costs to store**:

1. the 2-RDM -- one contraction away from the CI vectors that are still there, and the term
   that grows as the fourth power of the active space;
2. the CI vectors -- one CI solve at the stored orbitals. They are a Davidson warm start,
   worth an order of magnitude on the next solve and nothing after that;
3. the quasi-Newton curvature -- not regenerable at all, only discardable: a cold start at
   the same orbitals converges and pays a few extra macro-iterations for it. Each of those
   is a CI solve plus an integral transform, which is why it is last.

⚠ **A converged checkpoint inverts that and stores no curvature at all.** Nothing resumes a
converged optimization, and on a large basis the curvature scales with the rotation-parameter
count while everything else scales with the active space -- so it is most of the file
everybody keeps. What that file does need is the CI vectors, because they are what lets the
finished calculation be materialized without re-solving it from cold.

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
  the file would be refused rather than reconciled;
* the converged checkpoint reporting what it dropped -- ``curvature``, and nothing else;
* the materialized CASSCF reporting the run's own energy, and taking its state count from
  the file rather than from the call: there is no optimization left to configure, so the
  file's word is the only reading that can be right. ⚠ The one diagnostic that does *not*
  come back is the boundary check at the *starting* orbitals, because that is a statement
  about a trajectory and the trajectory is over;
* the perturbation's table -- eight scalars per (state, class) pair, so a few kilobytes
  whatever the active space, and no reference in it at all.

FINISHING A CALCULATION THAT IS ALREADY OVER
--------------------------------------------
``restart=`` resumes a running optimization; ``CASSCF.from_checkpoint`` materializes a
finished one. Both exist because re-entering the optimizer with nothing left to do ends its
table "NOT converged in 0 macro-iterations" on a calculation that converged hours earlier.

Nothing is optimized. The orbitals come back exactly and the states are re-solved at them,
seeded by the stored CI vectors -- which starts the eigensolver at the answer, so the cost is
one integral transform. ⚠ With the vectors thinned away the solve is cold: the *spectrum* is
the same, but inside a degenerate manifold the eigensolver's basis is arbitrary and this is a
different arbitrary basis. Nothing phase-invariant changes, which is everything a property
file is compared through, but the run says so rather than leaving it to be discovered.

SC-NEVPT2, CLASS BY CLASS
-------------------------
The correction loops over states and, inside that, over eight independent excitation
classes, and it is hours of work on a large reference. A job that dies in the last class
loses every class before it -- and the insurance is nearly free, because one class result is
eight scalars. The whole table is kilobytes whatever the active space, so unlike the CASSCF
file this one never has to choose what to give up; the only cadence knob is a minimum
interval, which rations filesystem calls rather than bytes.

⚠ **It stores no reference.** The orbitals and the CI vectors have one owner and it is the
CASSCF checkpoint; what this file keeps is a digest of them, and a restart against a
different reference is refused rather than merged into the table. The reason is physical:
inside a degenerate manifold the CI's basis is arbitrary and the per-state corrections carry
an arbitrary share of the manifold's internal spread, so resuming across a re-solved
reference would compute some members in one basis and some in another -- and the manifold's
barycentre, which is the quantity that means something, would belong to neither run.
"""
from __future__ import annotations

import math
import os
import shutil
import signal
from pathlib import Path
from typing import List

import numpy as np

import kuiva
from kuiva.io.checkpoint import read_checkpoint
from kuiva.pt.checkpoint import read_nevpt2_checkpoint
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
    # 5. Finishing a calculation that is already over.
    # ----------------------------------------------------------------------------------
    # ⚠ `restart=` resumes a *running* optimization; `from_checkpoint` materializes a
    # *finished* one. The difference is the whole reason both exist: re-entering the
    # optimizer with nothing left to do ends its table "NOT converged in 0 macro-iterations"
    # on a calculation that converged hours earlier.
    #
    # Nothing is optimized below. The orbitals come back from the file exactly; the states
    # are re-solved at them, seeded by the stored CI vectors, which starts the eigensolver
    # at the answer. That is why the converged checkpoint keeps its CI vectors and drops its
    # quasi-Newton curvature instead: nothing resumes a converged run, and on a large basis
    # the curvature is the majority of the file.
    out.section(log, "The finished CASSCF, read back from its file")
    with timing.timer("CASSCF (materialized)") as t_back:
        recovered = kuiva.CASSCF.from_checkpoint(straight_chk, reference).run()
    log.info("%s", recovered.summary())

    converged_chk = read_checkpoint(straight_chk)
    out.entries(log, [
        ("energy from the file", recovered.energy, "Eh", "", out.E_FMT),
        ("difference from the run", abs(recovered.energy - full.energy), "Eh", "",
         out.SCI_FMT),
        ("states re-solved at the stored orbitals", recovered.n_states, "",
         "n_states came from the file, not from this call"),
        ("thinned away by the converged write",
         ", ".join(converged_chk.dropped) or "nothing"),
        ("boundary at the starting orbitals", "not in the file", "",
         "it is a statement about a trajectory, and the trajectory is over"),
    ])

    # ----------------------------------------------------------------------------------
    # 6. The perturbation's own checkpoint.
    # ----------------------------------------------------------------------------------
    # SC-NEVPT2 loops over states and, inside that, over eight independent excitation
    # classes. A job that dies in the last class loses every class before it -- and the
    # insurance is nearly free, because one class result is eight scalars. The whole table
    # is kilobytes whatever the active space, so there is no budget to weigh here: unlike
    # the CASSCF file, this one never has to choose what to give up.
    #
    # ⚠ It stores no reference. The orbitals and the CI vectors have one owner and it is the
    # CASSCF checkpoint; what this file keeps is a *digest* of them, and a restart against a
    # different reference is refused rather than merged into the table. The reason is
    # physical: inside a degenerate manifold the CI's basis is arbitrary, so resuming across
    # a re-solved reference would compute some members in one basis and some in another, and
    # the manifold's barycentre would belong to neither run.
    out.section(log, "SC-NEVPT2, checkpointed by class")
    e2_chk = OUTPUT / "ticl3_nevpt2.chk"
    if e2_chk.exists():
        e2_chk.unlink()
    with timing.timer("NEVPT2 (first run)") as t_pt:
        correction = kuiva.NEVPT2(recovered, checkpoint=e2_chk).run()
    log.info("%s", correction.summary())

    table_on_disk = read_nevpt2_checkpoint(e2_chk)
    table_on_disk.report(log)
    out.entries(log, [
        ("checkpoint file", str(e2_chk.relative_to(HERE))),
        ("size", e2_chk.stat().st_size / 1024.0, "kB", "",
         "{:.1f}"),
    ])

    # A restart over a table that is already complete: every (state, class) pair is read
    # rather than evaluated. A real restart resumes a *partial* table the same way -- this
    # is that mechanism with nothing left to do, which is what makes it cheap to show.
    with timing.timer("NEVPT2 (restarted)") as t_pt_restart:
        resumed_pt = kuiva.NEVPT2(recovered, restart=e2_chk).run()

    # ⚠ `restart=` reads and `checkpoint=` writes, here as on the CASSCF stage. A resumed run
    # that is to keep protecting itself is given both, usually the same path.
    rebuilt = kuiva.NEVPT2.from_checkpoint(e2_chk, recovered, report=False).run()
    out.entries(log, [
        ("E2 of the lowest state", correction.e2[0], "Eh", "", out.E_FMT),
        ("restarted run, same state", resumed_pt.e2[0], "Eh", "", out.E_FMT),
        ("assembled from the file", rebuilt.e2[0], "Eh", "", out.E_FMT),
        ("first run", t_pt.wall, "s", "", "{:.2f}"),
        ("restart, evaluating nothing", t_pt_restart.wall, "s", "", "{:.2f}"),
    ])

    # ----------------------------------------------------------------------------------
    # 7. Side by side.
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
    # 8. Assert.
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
        # ⚠ Materialization is exact on the stored total: the energy is the file's number,
        # not one re-derived from re-solved states.
        "the materialized CASSCF reports the run's own energy":
            recovered.energy == full.energy,
        "the materialized CASSCF reports itself converged": bool(recovered.converged),
        "its state count came from the file": recovered.n_states == N_STATES,
        # ⚠ The inversion: a converged run is never resumed, so the file everybody keeps
        # carries the CI vectors and not the curvature.
        "the converged checkpoint dropped its curvature and kept its vectors":
            ("curvature" in converged_chk.dropped
             and converged_chk.ci_vectors is not None),
        "the perturbation's table is complete": table_on_disk.complete,
        # Nothing is recomputed on either route, so anything but exact equality would mean
        # the table is not what the run produced.
        "a restart over a complete table reproduces E2 exactly":
            bool(np.array_equal(correction.e2, resumed_pt.e2)),
        "the file assembles back into the same correction":
            bool(np.array_equal(correction.e2, rebuilt.e2)),
        # ⚠ The design claim, asserted against the bytes rather than against the docstring.
        "the perturbation checkpoint carries no reference (< 128 kB)":
            e2_chk.stat().st_size < 128 * 1024,
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
