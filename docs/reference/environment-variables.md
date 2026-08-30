# Environment variables

Every `KUIVA_*` variable in one place. Precedence, everywhere: an explicit argument to a
call overrides the environment, which overrides `defaults.conf`
([configuration](../guide/configuration.md)).

| variable | values | meaning |
|---|---|---|
| `KUIVA_MEMORY_GB` | GB, e.g. `32` | the memory limit for this job — what Kuiva may commit to its own arrays, not the machine's RAM. No built-in default: with no configured value either, a calculation refuses to start |
| `KUIVA_CONFIG` | a path | an explicit `defaults.conf`, for batch scripts; takes precedence over the whole search path |
| `KUIVA_NUM_THREADS` | an integer | the **total** thread budget; Kuiva spends it per stage (BLAS in dense stages, kernels in network sweeps). Fallback: `OMP_NUM_THREADS`, then every core the process may run on |
| `KUIVA_KERNELS` | `auto` (default), `numpy`, `native` | the kernel backend: compiled where available / pure NumPy (the reproducibility setting) / refuse if the build is missing or stale |
| `KUIVA_AMF_CACHE` | a directory, or `off` | the persistent atomic mean-field cache (default `~/.cache/kuiva/amf`); on a cluster, point it at a shared filesystem ([clusters](../guide/clusters.md#a-batch-script-skeleton)) |
| `KUIVA_SCRATCH` | a directory | the scratch directory. No built-in default and no `$TMPDIR`/cwd fallback: any scratch use refuses without one. A real disk, never a RAM-backed tmpfs |
| `KUIVA_SCRATCH_GB` | GB | the space Kuiva may use in the scratch directory |
| `KUIVA_CHECKPOINT_GB` | GB | the largest checkpoint file the adaptive budget may write |

**Scheduler variables are read only on request**: `$SLURM_JOB_ID` and
`$SLURM_JOB_END_TIME` are consulted only by `deadline="slurm"`/`"queue"`/`"auto"`
([clusters](../guide/clusters.md#stopping-before-the-queue-does-deadline)), never on
Kuiva's own initiative.

**Thread-library variables** (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, …) are respected in one
direction: a width the environment configured is left standing, and Kuiva pushes its budget
to the BLAS only when `KUIVA_NUM_THREADS` or an explicit call set it. Whether the BLAS
actually threads is measured at startup and stated in the banner
([configuration](../guide/configuration.md#threads-one-number)); `python -m
kuiva.util.threads` prints the numbers behind the verdict.
