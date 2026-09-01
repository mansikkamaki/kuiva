# Kuiva

Relativistic multireference quantum chemistry for strongly correlated, strongly relativistic
systems — 5d transition metals, lanthanides, actinides, and single-molecule magnets.

- **Two-component X2C relativity**, with spin–orbit coupling introduced at the CASSCF/CI
  level rather than at SCF, so the reference orbitals stay real and easy to converge.
- **Two-electron spin–orbit coupling by default** (atomic mean-field X2C, X2CAMF), which is
  what makes computed spin–orbit splittings quantitative rather than 5–30% too large.
- **Conventional complex determinant CI or an in-house tree tensor network** (DMRG/TTNS) for
  the CI step, sharing one state-averaged orbital optimizer.
- **SC-NEVPT2** for dynamic correlation, re-derived in spinor second quantization.
- **The deliverable is a file**: the effective Hamiltonian, the magnetic-moment and
  electric-dipole operators in the basis of the spin–orbit eigenstates, for an external
  crystal-field / ITO analysis code.

A calculation is a short **Python script**, not a text input file: one object per pipeline
stage, each built from the finished stage before it.

> **This is an early release, usable for production work with care.** Read
> [what not to trust](docs/limitations.md) before trusting a number.

## Documentation

**The manual is [`docs/`](docs/README.md)** — installation and configuration, running on
clusters, worked calculations from simple to hard, the working equations of every method
with full citations, and a complete option reference. Start with
[the documentation index](docs/README.md); this file is only the front door.

## Installation

Python ≥ 3.9, NumPy 1.x, SciPy, h5py, PySCF 2.14.0, `basis_set_exchange`, and NumPy on a
threaded BLAS. No compiler is needed — the compiled kernel backend is optional.

```bash
git clone https://github.com/mansikkamaki/kuiva.git
cd kuiva
python -m venv venv && . venv/bin/activate
pip install .
source setup.sh                     # once per shell — must be sourced, not executed
```

The first `source setup.sh` asks for a **memory limit** (Kuiva has none built in and never
guesses one) and a scratch directory. Details, including the optional Intel-only compiled
backend: [installation](docs/guide/installation.md) and
[configuration](docs/guide/configuration.md).

## A first calculation

Planar TiCl₃ — Ti(III), one 3d electron in a D₃ₕ ligand field — start to finish, a couple of
minutes on a laptop:

```python
import kuiva
from kuiva.util.logging import add_file_handler

add_file_handler("ticl3.out")

r = 2.25                               # Ti-Cl bond length in Angstrom
mol = kuiva.Molecule(
    atoms=[("Ti", (0.0, 0.0, 0.0)),
           ("Cl", (r, 0.0, 0.0)),
           ("Cl", (-0.5 * r, 0.8660254 * r, 0.0)),
           ("Cl", (-0.5 * r, -0.8660254 * r, 0.0))],
    basis="x2c-SVPall-2c", charge=0, spin=1)

scf = kuiva.ScalarSCF(mol, memory_gb=6.0).run()     # scalar X2C SCF: real orbitals
ref = kuiva.Reference(scf).run()                    # working basis, spinors, integrals

cas = kuiva.CASSCF(ref,
                   character=("Ti", "d"),           # the active space, stated as physics
                   n_active=10, n_active_elec=1,    # the 3d shell, one electron
                   n_states=2).run()                # averaged over the ground doublet

print(cas.summary())
kuiva.PropertyDump(cas, "ticl3.props", title="TiCl3 d1").run()
```

That is the whole shape of a Kuiva calculation. Every line — and the three decisions hiding
in it — is explained in [a first calculation](docs/guide/first-calculation.md); the harder
real cases (state-average design, double shells, broken symmetry, embedding) are in
[workflows](docs/guide/workflows.md).

## Examples and tests

`examples/` holds complete runnable calculations, one directory per pipeline stage — a
commented script plus the committed output of a known-good run; each asserts what it claims
and doubles as a smoke test:

```bash
source setup.sh
cd examples/01_scf_and_reference
python scf_and_reference.py
diff output/scf_and_reference.out reference/scf_and_reference.ref.out   # timings/version differ
```

```bash
pip install '.[test]'
pytest                     # the default suite: fast tests only, ~8 minutes
```

## Versioning

**Version 0.37.0.** `MAJOR.MINOR.PATCH`; while the leading digit is 0 the interface is not
yet stable and a breaking change moves `MINOR`. Every commit carries a version bump, so a
version identifies exactly one state of the code — the run banner prints it and every
stored product records it:

```python
import kuiva; kuiva.__version__          # '0.37.0'
```

Independent of it, checkpoints carry a `schema_version` and the plain-text products a
`format_version`, bumped only when the meaning or layout of something stored changes — see
[the stored products](docs/reference/files.md).

## Citing Kuiva

`CITATION.cff` at the repository root is the machine-readable citation; GitHub renders it
and citation managers read it directly. Please also cite the methods, basis sets and
libraries the calculation actually used: they are listed, numbered and linked in
[the bibliography](docs/references.md), and the `[PROVENANCE]` block of a property dump
names the Hamiltonian and basis you were actually running.

## Licence

Apache License 2.0; see `LICENSE` and `NOTICE`. The attribution obligation above is
independent of it: every method, algorithm and basis set implemented here is someone else's
published work and is cited as such in [the bibliography](docs/references.md).
