# Running on clusters

Kuiva targets **one fat node with shared memory**: threaded BLAS plus compiled kernels that
take their thread count as an explicit argument. There is no MPI and no distributed tensor
layer — memory, not core count, is the scaling limit, and the memory plan (see
[Configuration](configuration.md#the-memory-limit)) is what refuses a calculation that will
not fit before any time is spent on it.

What a cluster adds to the picture is that jobs get killed: by a queue time limit that is
known in advance, and by preemptions and drains that are not. This page covers surviving
both — checkpointing, `deadline=`, `signals=`, and restarting — plus the batch-script
plumbing.

## A batch-script skeleton

```bash
#!/bin/bash
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@600            # only if the script uses signals=, see below

source /path/to/kuiva/setup.sh

export KUIVA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KUIVA_MEMORY_GB=48              # leave headroom under --mem: the OS, PySCF's own
                                       # allocation, and everything Kuiva marks `external`
export KUIVA_SCRATCH=$LOCAL_SCRATCH    # a real disk, never a RAM-backed tmpfs
export KUIVA_AMF_CACHE=/proj/$USER/kuiva-amf   # shared: pay each atomic solve once ever

python calculation.py
```

Three of those lines are worth a sentence each:

- **`KUIVA_MEMORY_GB` is what Kuiva may commit to its own arrays, not the allocation.** The
  PySCF SCF allocates outside Kuiva's ledger and is reported as `external`, so a limit equal
  to `--mem` gets the job OOM-killed by the scheduler while Kuiva's own accounting is clean.
- **`KUIVA_AMF_CACHE` on a shared filesystem** means the four-component atomic solves behind
  the default spin–orbit treatment [[16]](../references.md#r16) — tens of minutes for a
  lanthanide — are paid once per project, not once per job. The cache is keyed on
  `(element, basis, configuration, interaction, nuclear model)`, so it is never silently
  reused for a different Hamiltonian.
- **The tensor-network solver wants a small thread width.** A DMRG sweep takes the same wall
  time on 1 and on 8 BLAS threads while spending several times the CPU on spin-wait, so give
  a network-heavy job fewer threads than a dense CASSCF and read its CPU-second figures
  accordingly.

## Checkpointing

```python
cas = kuiva.CASSCF(ref, checkpoint="run.h5", **space).run()
```

Checkpoints are schema-versioned HDF5 [[3]](../references.md#r3), written every
macro-iteration under an adaptive budget, with the converged one always written. They hold
the orbitals and orbital partition, the 1-RDM, the state energies, the run metadata, the
state average, and a fingerprint of the *system* — molecule, basis, nuclear model,
Hamiltonian axes, gauge origin — which a restart against the wrong file is **refused** on:
orbitals are an orthonormal set the optimizer will happily re-optimize from whatever they
came from, so without the fingerprint a restart on the wrong file converges to a plausible
number.

Four-index integrals are never stored, nor are transition densities: a restart regenerates
them, which is cheap because the expensive atomic solves have their own persistent cache. A
DMRG-CASSCF writes a second, sibling file (`*.network.h5`) with the network state, rolling
at the end of each completed sweep under the same cadence knobs; losing it costs a cold
first solve on restart — time, never correctness.

**The adaptive budget** (`checkpoint_budget_gb`, `checkpoint_min_interval_seconds`,
`checkpoint_cost_fraction` — in `defaults.conf` or per stage via `checkpoint_options=`)
estimates each write's size and time and skips or thins a write that would cost more than
~5% of the compute elapsed since the last one. Thinning gives up the large optional pieces
⚠ **in order of what it costs to get them back, not what they cost to store**: first the
2-RDM (one contraction away from the CI vectors that are still there), then the CI vectors
(one CI solve at the stored orbitals — a Davidson warm start worth an order of magnitude on
the next solve), and last the quasi-Newton curvature, which is not regenerable at all, only
discardable at the price of a few extra macro-iterations. ⚠ A **converged** checkpoint
inverts that and stores no curvature: nothing resumes a converged run, and on a large basis
the curvature is most of the file — while the CI vectors are what makes
`from_checkpoint` (below) cheap.

⚠ **Failure semantics are the opposite of a cache's, on purpose.** A checkpoint *write*
failure is a warning and the run continues; a *read* failure on an explicitly requested
restart is an error that propagates, because silently starting over wastes exactly the hours
the file existed to protect. A schema-version mismatch hard-refuses; a changed code
fingerprint only warns.

## Stopping before the queue does: `deadline=`

A checkpoint is what survives a job being killed. `deadline=` is how the job stops itself
first, while there is still time to write one.

```python
cas = kuiva.CASSCF(ref, checkpoint="run.h5", deadline="slurm", **space).run()
```

| `deadline=` | what it does |
|---|---|
| `None` *(the default)* | **no deadline.** Nothing is read, nothing is printed, nothing stops the run early |
| `"6h"`, `"90m"`, `"24:00:00"`, `21600` | a budget of your own, starting when the stage is *constructed* |
| `"slurm"` / `"queue"` | this batch allocation's own time limit — and **refuses** if it cannot be read |
| `"auto"` | that limit where there is one, no deadline where there is not, stated either way |

- ⚠ **No deadline is the default, and it is a decision rather than an omission.** A cluster
  with no queue time limit is an ordinary place to run; a deadline invented for such a run
  could only ever end it early for no reason. Nothing is read from the environment unless
  you ask for it.
- ⚠ **An explicitly named source that cannot be read refuses.** `deadline="slurm"` outside a
  Slurm job raises at once, naming both ways out. The alternative — quietly running with no
  deadline — is the one outcome worse than either: the job is killed at the wall twelve
  hours later and nothing in the output ever said the request had failed. `"auto"` is the
  spelling for a script that has to run on a laptop, on an unlimited cluster and inside a
  queue without being edited.
- **Where the limit comes from.** Slurm's is read once, at the start: from
  `$SLURM_JOB_END_TIME` (free, needs no client binaries, contacts nothing) and then from
  `scontrol show job`. ⚠ Once, never in a loop — polling the controller from every
  macro-iteration of every job is what makes a scheduler slow for everyone — so an
  allocation *extended* after the run started is not noticed, and the run stops at the limit
  it was told about. A job with no time limit (`EndTime=Unknown`) produces a deadline that
  never fires and says so. Another queue system is two functions and one registry entry in
  `kuiva.util.deadline`; nothing else in the program learns a scheduler's name.
- ⚠ **The decision is predictive, not reactive.** Stopping when the budget is already spent
  is stopping too late, because the checkpoint is written *afterwards*. The run stops when

  ```
  time left  <  longest recent macro-iteration  +  estimated checkpoint write  +  margin
  ```

  Every term but the last is measured — the iteration times from the optimizer's own table,
  the write from the measured disk bandwidth — and the margin (60 s) is printed with the
  rest. The final write is then **forced** past the cadence rules, exactly as a converged
  one is, because that checkpoint is the result and not insurance.
- ⚠ **A budget's clock starts when the stage is built, not when it is run**, so build the
  stage next to its `.run()`. A queue limit is an absolute instant and cannot drift this
  way.
- ⚠ **A bare numeric string is refused** — `"60"` is sixty *minutes* to Slurm's `--time` and
  would be sixty *seconds* here, and a factor of sixty in a deadline is what ends an
  allocation with nothing written. Write `60`, or `"60m"`.
- ⚠ **The granularity is one macro-iteration.** The run stops between them and nowhere else:
  a CI solve, a DMRG solve and an NEVPT2 excitation class cannot be interrupted, and if one
  macro-iteration outlives the allocation, no deadline can help. A run that has less than
  the margin left when it starts is refused rather than started. Without `checkpoint=` a
  deadline still stops the run cleanly, and says plainly that nothing was saved.

## The kill nobody announced: `signals=`

A deadline covers the case where the end is known in advance. `scancel`, a preemption, a
node draining and a `SIGTERM` at the wall are the case where it is not.

```python
cas = kuiva.CASSCF(ref, checkpoint="run.h5", deadline="slurm", signals=True, **space).run()
```

`signals=True` catches **`SIGTERM`, `SIGUSR1` and `SIGUSR2`**; a sequence names them instead
(`signals=("TERM", "INT")`). The run then stops at the next macro-iteration boundary with
its checkpoint written — the same stop, the same forced write, the same warning as a
deadline, differing only in what asked for it.

In a Slurm script, `--signal` is how you buy the lead time:

```bash
#SBATCH --signal=B:USR1@600     # SIGUSR1 ten minutes before the wall
```

⚠ The lead has to exceed one macro-iteration plus the checkpoint write, or the kill lands
first anyway. If the limit is knowable, `deadline="slurm"` is the better instrument: it
computes that reserve from what the run is actually doing instead of asking you to guess it.

Four things about `signals=` are deliberate:

- ⚠ **Off by default, and never otherwise.** A library that installs signal handlers behind
  your back breaks embedding, test runners and notebooks.
- ⚠ **The handlers are a loan.** They are installed for the duration of the stage and the
  previous dispositions are restored when it ends, exception or not. Off the main thread,
  where Python cannot install a handler at all, `signals=` is **refused** rather than
  silently skipped.
- ⚠ **A second signal is not waited for.** The first is a request; the second restores what
  was there before and re-raises, so the process dies exactly as it would have without Kuiva
  in it. `SIGINT` is not in the default set for the same reason — Ctrl-C is expected to
  interrupt *now*, not at the end of a macro-iteration.
- ⚠ **The request outlives the stage that caught it.** A signal is delivered to the process,
  so an `NEVPT2`, `CASCI` or `PseudospinExport` started afterwards raises
  `kuiva.util.signals.StopRequested` instead of beginning work the process will not live to
  finish. That is what makes a signalled run *exit* rather than merely pause:

  ```python
  from kuiva.util.signals import StopRequested
  try:
      pt = kuiva.NEVPT2(cas).run()
  except StopRequested:
      sys.exit(0)                       # the CASSCF's checkpoint is on disk
  ```

⚠ **What no signal can do is interrupt a solve.** A `SIGKILL` cannot be caught at all, and a
`SIGTERM` arriving inside one CI solve, one DMRG solve or one NEVPT2 class is acted on when
that finishes. The protection is the checkpoint cadence, not the handler.

## Restarting

Two different things restore a calculation from its file, and they default differently
because they answer different questions:

**`restart=` resumes a *running* optimization.** The caller restates the calculation and the
file *checks* it: the system fingerprint must match, the active space must agree, and
⚠ **a restart that changes the state average is refused** — `n_states` and `weights` are
recorded in the checkpoint, and a different average is a different *calculation* (the energy
functional itself changes), not a different chart of this one. Continuing from converged
orbitals into a **new** state average is a real thing to want, and it is `coeff=`, not
`restart=`. `max_iter` counts **total** macro-iterations across the restart, so an
interrupted-and-restarted run costs what an uninterrupted one would. (A checkpoint written
before the average was recorded cannot be checked, and says so rather than passing quietly.)

**`CASSCF.from_checkpoint(path, ref)` materializes a *finished* one**, with no
macro-iterations at all, so a dump or a correction that was never requested does not cost
the CASSCF again:

```python
cas = kuiva.CASSCF.from_checkpoint("run.h5", ref)   # nothing is optimized
kuiva.PropertyDump(cas.run(), "props.out").run()
```

The orbitals come back exactly and the states are re-solved at them, seeded by the stored CI
vectors, which starts the eigensolver at the answer. ⚠ With the vectors thinned away the
solve is cold: the spectrum is the same, but inside a degenerate manifold the eigensolver's
basis is arbitrary and this is a different one — nothing phase-invariant moves (see
[notation](../notation.md#degenerate-blocks-and-arbitrary-phases)), and the run says so.
`n_states`/`weights` come **from** the file here, because there is no optimization left to
configure. The one diagnostic that does not come back is the state-average boundary check at
the *starting* orbitals: it is a statement about a trajectory, and the trajectory is over.

**SC-NEVPT2 checkpoints too** (`NEVPT2(cas, checkpoint=path, restart=path)`), per
`(state, class)` pair — the only boundary that stage has. A class result is a handful of
scalars, so the whole table is kilobytes whatever the active space, and it never has to
choose what to give up; `deadline=` and `signals=` stop it between classes with the finished
ones written, and `NEVPT2.from_checkpoint(path, cas)` rebuilds the finished correction from
the table alone. ⚠ **It stores no reference** — the orbitals and CI vectors belong to the
CASSCF file, and this one keeps a digest. A restart against a different reference is
refused: inside a degenerate manifold the CI basis is arbitrary, so resuming across a
re-solved reference would compute some members of a manifold in one basis and some in
another, and the barycentre would belong to neither run.
