# Configuration

Kuiva's configuration surface is deliberately small: one file of site defaults, a handful of
environment variables, and one thread knob. Two values — the **memory limit** and the
**scratch directory** — have **no built-in default at all**, by design: whoever knows the
machine chooses them, once, rather than inheriting a number that fits someone else's
hardware. Everything else has a sensible default.

Precedence, everywhere: **an explicit argument to a call** overrides **the environment**,
which overrides **the configuration file**.

## The configuration file

Site defaults live in a `defaults.conf`, read from the first of these that exists:

| path | for |
|---|---|
| `$KUIVA_CONFIG` | an explicit file, for batch scripts |
| `~/.config/kuiva/defaults.conf` | per user — this is the one `setup.sh` writes |
| `/etc/kuiva/defaults.conf` | per machine (administrator) |
| `<sys.prefix>/etc/kuiva/defaults.conf` | per installation (administrator) |
| `<source tree>/defaults.conf` | a source checkout |

The distribution deliberately ships none of them, which is why the first `source setup.sh`
asks for the two defaultless values. Sections are for the reader's benefit and are flattened
when read; a malformed file is a hard error rather than a silent fallback.

```ini
[memory]
memory_gb = 32.0          # what Kuiva may commit to its own arrays. No default.
warn_fraction = 0.7       # warn above this fraction of the limit, still proceeding
allow_overcommit = false  # downgrade every memory refusal to a warning

[amf]
# amf_cache_dir = /scratch/$USER/kuiva-amf     # or "off"

[checkpoint]
# checkpoint_budget_gb = 2.0                   # largest checkpoint file
# checkpoint_min_interval_seconds = 0.0        # 0 = every macro-iteration
# checkpoint_cost_fraction = 0.05              # of the compute since the last one

[scratch]
# Like memory_gb, scratch_dir has NO built-in default and no $TMPDIR/cwd fallback: any
# scratch use (a factor spill, an out-of-core decomposition, network environment paging)
# refuses until it is set here or with $KUIVA_SCRATCH. Pick a real disk with room, never
# a RAM-backed tmpfs. Calculations that never touch scratch do not need it.
# scratch_dir = /scratch/$USER
# scratch_gb = 100.0
```

## Environment variables

| variable | meaning |
|---|---|
| `KUIVA_MEMORY_GB` | the memory limit for this job; overrides `defaults.conf` |
| `KUIVA_CONFIG` | an explicit configuration file |
| `KUIVA_NUM_THREADS` | the **total** number of threads the calculation may use |
| `KUIVA_KERNELS` | kernel backend: `auto` (default), `numpy`, or `native` |
| `KUIVA_AMF_CACHE` | directory for the atomic mean-field cache, or `off` |
| `KUIVA_SCRATCH` | scratch directory (no built-in default; scratch use refuses without one) |
| `KUIVA_SCRATCH_GB` | the space Kuiva may use in the scratch directory |
| `KUIVA_CHECKPOINT_GB` | largest checkpoint file |

Kuiva reads no *scheduler* variable unless it is asked to: `$SLURM_JOB_ID` and
`$SLURM_JOB_END_TIME` are consulted only by `deadline="slurm"`/`"queue"`/`"auto"` (see
[Running on clusters](clusters.md)), never on Kuiva's own initiative.

## The memory limit

A calculation refuses to start until a memory budget has been set once. With a limit in
hand, Kuiva refuses a calculation that cannot fit **before** it starts — printing the whole
allocation plan, which phase peaks, whether the machine could have taken it, and which knob
to change — instead of dying halfway through with the hours already spent.

```bash
export KUIVA_MEMORY_GB=32           # per job, e.g. from a batch script
```

or in `defaults.conf`, or per call:

```python
scf = kuiva.ScalarSCF(mol, memory_gb=32.0).run()
```

Three things to understand about what the number means:

- **The limit is what Kuiva may commit to *its own* arrays — not the machine's RAM.** Leave
  room for the operating system, for the PySCF SCF (whose allocation Kuiva does not govern
  and reports as `external`), and for whatever else the machine has to do. Because of that
  `external` share, the accounted total is a lower bound on the process's real memory use;
  the drift between the plan and the observed peak is checked automatically at process exit
  and warned about.
- **A refusal is a hard error, on purpose**, raised before the allocation. The sizing
  functions behind it are exact and never pad. `allow_overcommit = true` (or
  `allow_overcommit=True` per call) downgrades every refusal to a warning, for the case
  where you know better than the plan.
- **A reservation lives as long as the process.** Nothing watches the arrays, so a finished
  calculation's memory stays on the ledger until it is given back — and the *n*-th system in
  a loop would otherwise be refused against a limit its predecessors filled, with a refusal
  that reads as a machine that is too small. Scope each calculation:

  ```python
  from kuiva.util import resources

  for system in systems:
      with resources.calculation(system.label):    # released at the end of the block
          run(system)
  ```

  `resources.clear()` does the same for a driver whose structure does not suit a `with`
  block. Neither weakens the check itself, and a refusal that does involve leftover
  reservations says how much of the shortfall came from an earlier calculation.

## The scratch directory

Like the memory limit, `scratch_dir` has **no built-in default and no `$TMPDIR`/cwd
fallback**: a guessed location lands on an unvetted filesystem, and a RAM-backed `/tmp`
would spend exactly the memory a spill exists to save. Any scratch use — an integral-factor
spill (`factors="scratch"`), an out-of-core Cholesky decomposition (`factors="streamed"`),
tensor-network environment paging — refuses with the knob named until `scratch_dir` (or
`$KUIVA_SCRATCH`) is set. Calculations that never touch scratch do not need it.

Pick a real disk with room — on a cluster, the node-local or parallel scratch filesystem,
never a RAM-backed tmpfs.

## Threads: one number

```bash
export KUIVA_NUM_THREADS=4        # the total this calculation may use
```

That is the whole knob. Kuiva then spends it where the stage needs it — on the threaded BLAS
in a dense stage such as the integral transform, and on the compiled kernels in a
tensor-network sweep, where a threaded BLAS measurably buys nothing and is charged for the
spin-wait anyway. Run the network solver at a small thread width and read its CPU-second
figures accordingly. The precedence is: explicit argument > `KUIVA_NUM_THREADS` >
`OMP_NUM_THREADS` > every core the process is allowed to run on.

At startup Kuiva **measures** whether the BLAS in the process actually threads at the width
you asked for, and prints the verdict in the banner:

```
             threads: 4 (KUIVA_NUM_THREADS); BLAS: MKL 2025.1, threaded
```

It warns if the answer is no — a budget that buys no threads spends CPU-hours for nothing —
and it warns in the other direction too, when a second threading runtime is found spinning
next to the BLAS's, because that silently inflates every timing the run reports. A BLAS with
no thread control at all is reported as "unverified" rather than as either verdict.

**Three BLAS libraries are controlled** — MKL [[5]](../references.md#r5), OpenBLAS
[[6]](../references.md#r6) and BLIS [[7]](../references.md#r7) — each bound at run time to
whichever one is actually mapped into the process, including the OpenBLAS a NumPy or SciPy
wheel brings with it. So the knob, the per-stage widths and the startup measurement work the
same way on a machine without Intel's BLAS, which is most clusters. ⚠ One difference is
worth knowing: MKL's width is per-thread, while OpenBLAS's and BLIS's is process-wide, so on
those a stage's width is visible to the whole process until that stage ends. To see the
numbers behind the verdict:

```bash
python -m kuiva.util.threads
```

## The kernel backend

| `KUIVA_KERNELS` | meaning |
|---|---|
| `auto` *(default)* | compiled where available, NumPy otherwise; a missing build is not an error |
| `numpy` | pure NumPy even if the build exists — the reproducibility setting |
| `native` | refuse to start if the build is missing or stale (an explicit request is never quietly downgraded) |

Checkpoints record which backend produced them, and a non-default backend announces itself
in the run banner.

## The atomic mean-field cache

The default spin–orbit treatment (X2CAMF [[16]](../references.md#r16)) costs one
four-component atomic solve per unique element — under a second for a light atom, tens of
minutes for a lanthanide. The solve depends on no geometry, so it is paid **once ever** per
`(element, basis, configuration, interaction, nuclear model)` and cached both in the process
and on disk: `~/.cache/kuiva/amf` by default, or the directory named by `$KUIVA_AMF_CACHE` /
`amf_cache_dir`, or `off` to disable the disk cache. A potential-energy surface pays the
solve once, not once per point. On a cluster, point the cache at a shared filesystem so
every job of a project reuses the same solves (see [Running on clusters](clusters.md)).

## Verbosity

Output verbosity is per subsystem, through the standard `logging` module:

```python
from kuiva.util.logging import set_verbosity
set_verbosity("DEBUG")            # per-micro-iteration detail; "TRACE" for shapes and paths
```

The default (`INFO`) reads like a conventional quantum-chemistry output file: a banner,
sections, label/value blocks and fixed-width iteration tables, ASCII only. `WARNING` and
`ERROR` lines are prefixed `*** WARNING [subsystem]` — they deliberately break the visual
flow and are greppable. `add_file_handler(path)` mirrors the stream to a file.
