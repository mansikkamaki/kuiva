# Installation

Kuiva is a Python package. No compiler is needed: the compiled kernel backend is optional,
and the pure-NumPy path is a first-class, fully supported way to run the code — not a
fallback.

## Requirements

- **Python ≥ 3.9**
- **NumPy ≥ 1.22, < 2** (NumPy 1.x — the cap is deliberate and matches the 3.9 floor)
- **SciPy ≥ 1.6**
- **h5py ≥ 3** (checkpoints and caches are HDF5 [[3]](../references.md#r3))
- **PySCF == 2.14.0** [[1]](../references.md#r1) — pinned, because the X2C front-end API has
  changed across PySCF releases and Kuiva is tested against exactly this one
- **basis_set_exchange ≥ 0.11** [[9]](../references.md#r9)
- NumPy on a **threaded BLAS** (MKL, OpenBLAS or BLIS — all three are recognized and
  thread-controlled; see [Configuration](configuration.md))

**All-electron basis sets only.** X2C has no meaning with an effective core potential, and
Kuiva refuses an ECP basis rather than pretend otherwise.

## Installing

```bash
git clone https://github.com/mansikkamaki/kuiva.git
cd kuiva
python -m venv venv && . venv/bin/activate
pip install .                       # or `pip install -e .` to work on the source
source setup.sh                     # once per shell — must be SOURCED, not executed
```

## What `setup.sh` does

`setup.sh` prepares the shell and then gets out of the way. It does three things:

1. **Checks the interpreter on `PATH`** — Python version, NumPy 1.x, SciPy, h5py, the
   pinned PySCF, a working BLAS — and reports *every* problem it finds at once rather than
   the first, so fixing an environment is one round trip. The verdict is cached, keyed on a
   fingerprint of the interpreter, so sourcing it again costs nothing and an upgraded
   interpreter is re-probed.
2. **Puts the repository root on `PYTHONPATH`**, so a run uses *this* tree rather than
   whatever copy happens to be installed elsewhere.
3. **Makes sure a memory limit has been configured**, asking for one exactly once and
   writing it to `~/.config/kuiva/defaults.conf`. Kuiva has **no built-in memory limit and
   never guesses one** — a calculation refuses to start until a budget has been chosen by
   someone who knows the machine. It asks for the scratch directory beside it, which has no
   default for the same reason. Both are explained in [Configuration](configuration.md).

It deliberately sets **no thread count**: that is `KUIVA_NUM_THREADS`, and a setup script
choosing a width silently would make every run's cost figures a property of the script.

## The optional compiled backend

A handful of measured-hot kernels — the determinant connection scan, the tensor-network
block GEMM and its sparse-operator sibling — have C++ implementations behind the same
registry the NumPy ones live in, bound through pybind11 [[4]](../references.md#r4). If you
have the Intel oneAPI toolchain:

```bash
cd cpp && ./configure && make && make check   # writes kuiva/_native<...>.so, git-ignored
```

`configure` **refuses any toolchain other than Intel oneAPI**, and says so plainly: the
extension has to share one MKL [[5]](../references.md#r5) and one OpenMP
[[8]](../references.md#r8) runtime with the interpreter that loads it, and a GCC build would
silently put a second OpenMP runtime next to MKL's — oversubscription with no error message.
That refusal costs nothing, because `KUIVA_KERNELS=numpy` is a supported way to run, not a
degraded one.

Parity between the backends is not a hope: a serial compiled kernel reproduces the NumPy one
**bit for bit**, and the only kernels allowed a tolerance (1e-13 relative) are those whose
docstring names the reduction a threaded run reorders. Which backend produced a result is
recorded in every checkpoint, and a run that did not use the plain NumPy path says so in its
banner — an unmarked output is by definition a pure-NumPy one. The backend is selected by
one environment variable, `KUIVA_KERNELS` (see
[Configuration](configuration.md#the-kernel-backend)).

## Checking the installation

```bash
source setup.sh
pip install '.[test]'
pytest                     # the default suite: fast tests only, budgeted at ~8 minutes
```

The default suite is laptop-fast and needs nothing beyond Kuiva's own dependencies. Two
further tiers are opt-in: `pytest -m slow` runs the slow tests (heavy-atom four-component
solves; hours), and `pytest -m ''` runs everything. The external programs that generated the
committed cross-check reference data (OpenMolcas [[51]](../references.md#r51), DIRAC) are
needed only to *regenerate* it; the suite as run compares against the stored files and never
re-runs them.

The `examples/` directory holds complete runnable calculations, one per pipeline stage, each
with the committed output of a known-good run; every example asserts what it claims and
exits non-zero on failure, so each doubles as a smoke test:

```bash
cd examples/01_scf_and_reference
python scf_and_reference.py                    # writes output/scf_and_reference.out
diff output/scf_and_reference.out reference/scf_and_reference.ref.out
```

In that `diff`, the timings, the memory summary and the banner's version and thread lines
are expected to differ; the physics is not. The committed references were generated on the
pure-NumPy path (`KUIVA_KERNELS=numpy`), so with a compiled backend built, quantities at the
1e-13 level may differ in their last digits — also expected.

## Next

[Configuration](configuration.md) — the memory limit, the scratch directory, threads and
the environment variables — and then [a first calculation](first-calculation.md).
