# Kuiva

Relativistic multireference quantum chemistry for strongly correlated, strongly relativistic
systems — 5d transition metals, lanthanides, actinides, and single-molecule magnets.

- **Two-component X2C relativity**, with spin–orbit coupling introduced at the CASSCF/CI level
  rather than at SCF, so the reference orbitals stay real and easy to converge.
- **Two-electron spin–orbit coupling by default** (atomic mean-field X2C, X2CAMF), which is what
  makes computed spin–orbit splittings quantitative rather than 5–30 % too large.
- **Conventional complex determinant CI or an in-house tree tensor network** (DMRG/TTNS) for the
  CI step, sharing one state-averaged orbital optimizer.
- **SC-NEVPT2** for dynamic correlation, over all eight excitation classes.
- **The deliverable is a file**: the effective Hamiltonian and the three magnetic-moment
  operators in the basis of the spin–orbit eigenstates, for an external crystal-field / ITO
  analysis code. Kuiva does not do that analysis itself.

A calculation is a short **Python script**, not a text input file: one object per pipeline
stage, each built from the finished stage before it.

> **This is an early release, usable for production work with care.** It is complete and
> tested along the path described below, and there are things in it that are approximations of
> unmeasured size, and sizes it cannot reach. Read [Limitations](#limitations--what-not-to-trust)
> before trusting a number.

**Contents** — [Installation](#installation) · [A first calculation](#a-first-calculation) ·
[The stages](#the-stages) · [The Hamiltonian](#the-relativistic-hamiltonian-what-each-option-means) ·
[Spin–orbit coupling](#spinorbit-coupling) · [Active spaces](#choosing-and-inspecting-an-active-space) ·
[Configuration](#configuration) · [Output files](#what-a-run-writes) ·
[Examples](#examples) · [Tests](#tests) · [Limitations](#limitations--what-not-to-trust) ·
[Versioning](#versioning) · [Citing](#citing-kuiva) · [References](#references)

---

## Installation

**Requirements:** Python 3.9 or newer, NumPy 1.x, SciPy, `h5py`, PySCF 2.14.0,
`basis_set_exchange`, and NumPy on a threaded BLAS. No compiler is needed: the compiled kernel
backend is optional, and the pure-NumPy path is a first-class way to run the code.

```bash
git clone https://github.com/mansikkamaki/kuiva.git
cd kuiva
python -m venv venv && . venv/bin/activate
pip install .                       # or `pip install -e .` to work on the source
source setup.sh                     # once per shell — must be sourced, not executed
```

`setup.sh` does three things and then gets out of the way:

- it **checks the interpreter** on `PATH` — Python version, NumPy 1.x, SciPy, `h5py`, the
  pinned PySCF, a working BLAS — and reports *every* problem it finds at once rather than the
  first. The verdict is cached per interpreter, so sourcing it again costs nothing;
- it puts the repository root on `PYTHONPATH`, so a run uses *this* tree;
- it makes sure a **memory limit** has been configured, asking for one exactly once (see
  below) and writing `~/.config/kuiva/defaults.conf`.

It deliberately sets **no thread count**: that is `KUIVA_NUM_THREADS`, and a setup script
choosing a width silently would make every cost figure a property of the script.

**All-electron basis sets only.** X2C has no meaning with an effective core potential, and
Kuiva refuses an ECP basis rather than pretend otherwise.

### The memory limit, which has no default

A calculation refuses to start until a memory budget has been set once. This is deliberate:
there is no built-in number and no auto-detection, because whoever knows the machine should
choose it — and with a limit in hand Kuiva refuses a calculation that cannot fit *before* it
starts, printing the whole allocation plan, which phase peaks and which knob to change,
instead of dying halfway through.

```bash
export KUIVA_MEMORY_GB=32           # per job, e.g. from a batch script
```

or in `defaults.conf` (see [Configuration](#configuration)), or per call:

```python
scf = kuiva.ScalarSCF(mol, memory_gb=32.0).run()
```

The limit is what Kuiva may commit to *its own* arrays — not the machine's RAM. Leave room for
the operating system, for the PySCF SCF (whose allocation Kuiva does not govern and reports as
`external`), and for whatever else the machine has to do.

### The optional compiled backend

A handful of measured-hot kernels — the determinant connection scan, the tensor-network block
GEMM and its sparse-operator sibling — have C++ implementations behind the same registry the
NumPy ones live in. If you have the Intel oneAPI toolchain:

```bash
cd cpp && ./configure && make && make check   # writes kuiva/_native<...>.so, git-ignored
```

`configure` refuses any toolchain other than Intel oneAPI, and says so plainly: the extension
has to share one MKL and one OpenMP runtime with the interpreter that loads it, and a GCC build
would silently put a second OpenMP runtime next to MKL's — oversubscription with no error
message. That refusal costs nothing; `KUIVA_KERNELS=numpy` is a supported way to run, not a
fallback.

Parity is not a hope: a serial compiled kernel reproduces the NumPy one **bit for bit**, and
the only kernels allowed a tolerance (1e-13 relative) are those whose docstring names the
reduction a threaded run reorders. Which backend produced a result is recorded in checkpoints,
and a run that did not use the plain NumPy path says so in its banner.

---

## A first calculation

Planar TiCl₃ — Ti(III), one 3d electron in a D₃ₕ ligand field — start to finish. It takes a
couple of minutes on a laptop.

```python
import kuiva
from kuiva.util.logging import add_file_handler

add_file_handler("ticl3.out")          # the run's output, as well as the terminal

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
print(cas.energies)                                 # total state energies [Eh]

kuiva.PropertyDump(cas, "ticl3.props", title="TiCl3 d1").run()
```

That is the whole shape of a Kuiva calculation. Three things in it are decisions you will have
to make in any real run, and they are the reason the API asks for them explicitly:

1. **The active space is stated as orbital character, not as indices.** *"The five lowest
   Kramers pairs of d character on the titanium"* is a definition another program can
   reproduce; *"spinors 68 to 77"* is not, and it silently follows the basis set around. A
   principal quantum number is refused on purpose: PySCF's principal-quantum AO labels count
   shells *within the basis*, which is not the same thing.
2. **Which states the orbitals are optimized for.** The state average here is the ground
   Kramers doublet, because that is the state whose orbitals are wanted. Averaging over
   everything would optimize the orbitals for an average nobody asked about.
3. **Where the state average stops.** A state-averaged CASSCF is exactly as symmetric as the
   set it averages over. A count that stops *inside* a near-degenerate manifold makes the
   averaged density non-invariant, the Fock operator built from it splits the shell, and the
   result is entirely plausible and wrong. The only evidence is a root the average does *not*
   use, so Kuiva solves a few extra, discards them, and reports the gap — at the starting
   orbitals as well as at the converged ones, since the starting one is what says whether the
   trajectory was safe. Below 50 cm⁻¹ it warns. The diagnostic never kills a run: a failure to
   measure the gap is a warning and no report, which is a weaker statement than a clean one.

   A clean gap is necessary and not sufficient, and the case to know about is a free ion: one
   complete `2J+1` manifold is a boundary, and it is still not an ensemble the symmetry of the
   *term* leaves invariant. Averaging a single J makes the spherical point a saddle of the
   function being optimized, so the run slides off it by however far rounding noise pushes —
   on a p¹ atom the residual g anisotropy of a j = ½-only average differed sixtyfold between
   two BLAS libraries given identical input, while the average over the whole term gave the
   analytic Landé values on both. Average whole terms; if a sub-manifold average is what you
   want, check the manifold splitting of the result directly.

Everything the run prints goes through one output grammar, so an output file reads like a
conventional quantum-chemistry output: a banner, sections, label/value blocks and fixed-width
iteration tables, ASCII only. `WARNING` and `ERROR` lines are prefixed `*** WARNING
[subsystem]` — they deliberately break the visual flow and are greppable. Verbosity is per
subsystem through the standard `logging` module:

```python
from kuiva.util.logging import set_verbosity
set_verbosity("DEBUG")                  # per-macro-iteration detail; "TRACE" for shapes/paths
```

---

## The stages

| stage | built from | what it does | headline results |
|---|---|---|---|
| `ScalarSCF` | a `Molecule` | scalar-relativistic X2C SCF (RHF / ROHF / UHF), integral ingestion, two-component spin–orbit Hamiltonian | `.energy`, `.converged`, `.data` |
| `Reference` | `ScalarSCF` | orthonormal working basis, Kramers-paired spinor guess, factorized two-electron integrals | `.reference`, `.nspinor` |
| `CheapCI` *(optional)* | `Reference` | cheap selected-CI pre-optimization: physical active orbitals, entanglement | `.orbitals`, `.occupations`, `.entropy`, `.mutual_information` |
| `CASSCF` | `Reference` or `CheapCI` | state-averaged two-component CASSCF, `solver="ci"` or `"dmrg"` | `.energy`, `.energies`, `.coeff`, `.converged` |
| `NEVPT2` *(optional)* | `CASSCF` | SC-NEVPT2, per state, by excitation class | `.e2`, `.total_energies`, `.class_energies` |
| `PropertyDump` | `CASSCF` or `NEVPT2` | the property-matrix file: `H`, `mu_x`, `mu_y`, `mu_z` | `.matrices`, `.path` |
| `PseudospinExport` | `CASSCF` | local multiplets, `H_eff` and moments on a pseudospin product basis | `.model`, `.g_values`, `.path` |

**One contract, obeyed by all of them** — learn one, guess the rest:

- the **constructor takes the finished upstream stage** plus keyword options, and validates
  everything it can immediately. A misspelled option, an impossible active space or a missing
  prerequisite fails at construction, not an hour into the run;
- **`.run()` is the only expensive call.** It executes the stage, stores the results as plain
  attributes and returns `self`, so `cas = CASSCF(ref, ...).run()` reads linearly. Calling it
  twice returns the same object without recomputing;
- **`.summary()`** returns a short plain-text block of the headline results;
- results are plain attributes, and the low-level objects stay reachable (`.data`,
  `.reference`, `.outcome`, `.result`). This layer is a thin wrapper **over**
  `kuiva.interface.api` and the module drivers, which remain public, unchanged and usable
  directly — `spinor_reference`, `casscf`, `sc_nevpt2`, `optimize_orbitals` and the rest are
  where you go to drive one piece of the pipeline by hand.

### `ScalarSCF(molecule, **options)`

Options are those of `kuiva.interface.api.scalar_x2c_reference`, validated by name:
`method` (the Hamiltonian, `"X2C-AMF"` by default), `x2c_approx`, `screening`,
`screening_options`, `decoupling_options`, `reference` (`"auto"`, `"rhf"`, `"rohf"`, `"uhf"`),
`memory_gb`, `gauge_origin`, `auxbasis`, `conv_tol`, `max_cycle`.

`reference="auto"` picks RHF for a closed shell and ROHF otherwise; RHF on an open shell is
refused rather than silently promoted. The gauge origin (default: centre of mass) is fixed
here, not at the property dump, because `L` is defined relative to it and the multireference
layer never calls PySCF again.

### `Reference(scf, **options)`

`threshold` (overlap-eigenvalue cutoff for linear-dependence removal, default 1e-7), `scheme`
(`"canonical"`, the default and the only one that can drop a dimension), `cholesky_tol`
(default 1e-8), `orbit_pivots`.

Dropping a vector is a deliberate, reported reduction of the variational space — the summary
says how many went. Two-electron integrals go through a **Cholesky decomposition by default**,
whose error is a threshold you set, rather than density fitting, whose error is not bounded at
all. The pivots are **complete symmetry orbits** (whole shell pairs), so the factorization
stays exactly invariant under rotations of an atom independently of the threshold; plain column
pivoting splits degeneracies that symmetry makes exact, which is a qualitative failure and not
a small one. Density fitting is fully supported on request — supply an auxiliary basis, and the
accuracy of that fit is then yours to judge. A Coulomb-fitting (J) auxiliary is *not* accurate
enough for correlated integrals, by orders of magnitude, and Kuiva warns if you ask for one.

```python
ref = kuiva.Reference(scf, cholesky_tol=1e-6).run()                  # a looser threshold
scf = kuiva.ScalarSCF(mol, auxbasis="def2-universal-jkfit").run()    # density fitting instead
scf = kuiva.ScalarSCF(mol, reference="uhf").run()                    # unrestricted scalar guess
```

⚠ An unrestricted spinor set is orthonormal but **not Kramers paired**. With `reference="uhf"`
an active space may therefore not be chosen as a contiguous spinor range; select by orbital
character per spin set.

### `CheapCI(reference, ...)` — optional pre-optimization

A cheap selected CI that rotates the raw spinor guess towards physical active orbitals. Its two
products feed the stages after it: the rotated orbitals start the CASSCF, and the entanglement
data seeds a tensor-network topology. A `CASSCF` built on a `CheapCI` inherits both, plus its
active space, unless told otherwise.

```python
pre = kuiva.CheapCI(ref, character=("Ti", "d"), n_active=10, n_active_elec=1).run()
pre.suggested_active()        # fractionally occupied spinors — a LOWER BOUND, not an answer
pre.dmrg_ordering()           # Fiedler ordering for a path network
```

⚠ **The pre-optimizer's total energy means nothing** and is deliberately not an attribute. What
it claims is that the *occupations* converge long before the energy does, which is what makes
it useful for selecting orbitals. And `suggested_active()` is a lower bound by construction:
occupation-based selection cannot flag an empty orbital that a better treatment would populate.
Combine it with orbital character and near-degeneracy.

### `CASSCF(upstream, ...)`

The active space: `character=(atom, l)` with `n_active=` (and optionally `n_active_elec=`), a
list `character=[(atom, l, n_spinors), ...]` to union per-fragment selections across centres,
or `active=[spinor indices]` for an explicit one. `atom` may be a sequence of centres, whose
populations are pooled — the right form for equivalent centres whose canonical orbitals
delocalize. There is no default: an active space is a physical statement.

The states: `n_states`, `weights`. The optimizer: `mode` (`"auto"`, `"quasi-newton"`,
`"second-order"`), `max_iter`, `conv_grad`, `conv_energy`, `max_step`, `callback`.
Checkpointing: `checkpoint=path`, `restart=path`, `checkpoint_options`.

`mode="auto"` is the robust default rather than the cheapest one, and escalates on the
*gradient* trajectory. ⚠ **It is the orbital problem that decides the mode, not the CI cost**:
`mode="second-order"` is the right explicit choice for a heavy element, a large state average
or a DMRG solver. That is a caller decision and is deliberately not inferred. If you set it,
give `max_iter` room to survive the escalation delay.

`solver=` picks the CI method behind the **same** orbital optimizer:

```python
cas = kuiva.CASSCF(ref, character=("Er", "f"), n_active=14, n_active_elec=11,
                   n_states=16, mode="second-order", checkpoint="er.h5").run()

cas = kuiva.CASSCF(pre, solver="dmrg", n_states=4, graph="mutual-information",
                   solver_options=dict(max_bond=128, adaptive=True)).run()
```

- **`"ci"`** (default) — conventional complex determinant CI, with checkpoint/restart and the
  state-average boundary diagnostic at both ends.
- **`"dmrg"`** — the in-house tree tensor network. `solver_options` must carry `max_bond` (an
  uncapped tree state allocates charge-sector-maximal bonds) and may carry `adaptive=True`,
  which routes the optimization through the event-gated driver so that network-topology changes
  are adopted only when they lower the energy at fixed integrals. `graph=` seeds the topology:
  a `NetworkGraph`, or `"mutual-information"` / `"fiedler"` to build one from a `CheapCI`
  upstream. ⚠ Checkpoint/restart of the network state is not wired into this layer.

A third solver, `kuiva.qc`, runs the configuration selection of a CI on a quantum computer or
its simulator through the same seam, and can drive a whole CASSCF on Kuiva's own exact simulator
or on Qiskit Aer. It is a **research vehicle**, never a default, optional, and dependent on
nothing the rest of the program needs. ⚠ The published algorithms it builds on assume a
spin-separable, real, non-relativistic Hamiltonian, which is none of what Kuiva has; the
generalization to complex spinor excitation generators is implemented and exact, as are a
sample-based Krylov variant, a variational quantum eigensolver, and a time-reversal-aware
configuration recovery with no counterpart in the spin-separable literature. **Exact is not the
same as good**: whether such an ansatz concentrates its measurements where they matter, at a
size where the question is interesting, needs hardware that is not yet reachable. So no number
from this layer is evidence about the *method*, and none is presented as one. Its references are
at the end of this file.

A restart takes its active space **from the file**; restating it in a way that disagrees is
refused rather than reconciled, and `max_iter` counts total macro-iterations across the
restart, so an interrupted-and-restarted run costs what an uninterrupted one would.

### `NEVPT2(casscf, **options)`

Strongly contracted NEVPT2 on a converged reference — post-processing, per state, decomposed by
excitation class. It consumes the converged orbitals and CI vectors and changes no
wavefunction. Options: `frozen_core`, `deleted_virtual`, `shift`, `imaginary_shift`, `fock`,
`classes`.

```python
pt = kuiva.NEVPT2(cas, frozen_core=-10.0).run()   # an orbital ENERGY, never a count
pt.multiplets()                                   # barycentres beside the per-state energies
```

- **Frozen core and deleted virtuals are off by default** and are stated as an orbital energy
  on the pseudo-canonical spectrum, never as a count.
- Intruder-state level shifts (real and imaginary) exist and are **parameter-free by default**;
  any applied shift warns.
- All eight excitation classes are a *partition* of the first-order interacting space, so a
  restricted `classes=` gives a **partial** `E2` and says so.
- ⚠ **Inside a degenerate CI manifold the individual per-state `E2` depend on the
  eigensolver's arbitrary basis; the barycentre does not.** No contraction fixes this, so the
  treatment is reporting rather than repair: `multiplets()` gives barycentres *beside* the
  per-state energies, with the member spread visible.
- Needs a `solver="ci"` CASSCF: a tensor-network reference has no stored CI vectors.

### `PropertyDump(source, path, ...)` and `PseudospinExport(casscf, path, ...)`

The two formatted products; see [What a run writes](#what-a-run-writes) for the files
themselves.

```python
kuiva.PropertyDump(cas, "ticl3.props", title="TiCl3 d1").run()
kuiva.PropertyDump(pt, "ticl3.props").run()      # NEVPT2-corrected H; hybrid protocol recorded

kuiva.PseudospinExport(cas, "ticl3.psd", sites=[tuple(range(10))],
                       rule="dimension", dims=2).run()
```

Passing a finished `NEVPT2` to `PropertyDump` substitutes the corrected energies on the
diagonal **and records the hybrid protocol in the header** (`H` from perturbation theory, `mu`
from the CASSCF states). That substitution is available only through this argument, never as a
flag, so the file and its provenance cannot be separated.

`PseudospinExport` partitions the **active spinors** into local multiplet sites (`sites=`, by
position in the active list; `None` discovers them from the converged state's entanglement) and
`rule`/`dims` choose each site's multiplet space — `rule="dimension"` with `dims=2` is *"the
ground Kramers doublet per site"*. ⚠ Every site must sit in **one particle-number sector**,
because a pseudospin labels a multiplet; a single delocalized electron over several sites is
refused. For a single centre, one site holding the whole active space is the right form.

---

## The relativistic Hamiltonian: what each option means

Kuiva is a two-component X2C code. Pick a Hamiltonian **by name**:

```python
scf = kuiva.ScalarSCF(mol)                             # X2C-AMF, the default
scf = kuiva.ScalarSCF(mol, method="X2C-AMF-DLU")       # the cheap end of the ladder
```

| method | one-electron decoupling | two-electron picture change | use it for |
|---|---|---|---|
| **`X2C-AMF`** *(default)* | exact, molecular | atomic mean field | **everything** |
| `X2C-AMF-DLU` | local (atom-blocked, DLU) | atomic mean field | systems where the exact decoupling is prohibitive |
| `X2C-1e` | exact, molecular | none | SOC-free comparisons, diagnostics |
| `X2C-1e-DLU` | local (atom-blocked, DLU) | none | diagnostics; doubly approximate |
| `X2C-mmf` | exact, molecular | **molecular** mean field | ⚠ benchmark only — see below |

A name resolves to two independent axes, which may also be set directly. Setting both a name
and an axis that contradicts it is refused rather than reconciled; a valid combination with no
canonical name is synthesized, so the provenance is never empty.

```python
kuiva.ScalarSCF(mol, x2c_approx="1e", screening="x2camf").run()   # = method="X2C-AMF"
kuiva.ScalarSCF(mol, method="X2C-AMF", screening="none").run()    # ValueError: contradictory
```

⚠ Selecting a **DLU** method emits a warning, and the choice is written into the provenance.
DLU is an approximation to the X2C transformation itself, offered for systems where the exact
decoupling will not fit; it is not a cheaper default.

**A calculation uses two different one-electron operators, deliberately.** The SCF that
produces the starting orbitals runs with the *spin-free* X2C operator; the correlated
(multireference) step, whose expectation value is the energy, uses the *full two-component*
one. Orbitals are a basis and are re-optimized later, so the scalar set is a guess — but the
two operators differ by a picture-change effect, and `soc.picture_change_shift` reports how
much.

| stage | one-electron operator | spin–orbit | two-electron picture change |
|---|---|---|---|
| SCF (orbital guess) | `sfx2c1e`, exact, molecular — **always**, unaffected by both knobs | no | no |
| correlated Hamiltonian | full 2c X2C, selected by `x2c_approx` | yes | selected by `screening` |

### `x2c_approx` — how the one-electron decoupling is built

| value | what it does | when to use it |
|---|---|---|
| `"1e"` *(default)* | Exact molecular one-electron X2C. `X` and `R` come from the full molecular four-component one-electron problem, via PySCF. | everything |
| `"1e-dlu"` | **DLU**: `X` *and* `R` are block-diagonal over atoms, so the transformation is local. Every molecular block is still transformed, including the off-diagonal ones. One small eigenproblem per atom instead of one of dimension `4·nao`. | when the exact decoupling will not fit |
| `"atom1e"` | `X` is built block-diagonally from **isolated-atom** one-electron problems; `R` is still the molecular one. ⚠ This is *not* DLU, and it is not cheaper in scaling — the `O(nao³)` work stays in `R`. | making the one- and two-electron decouplings consistent |

`decoupling_options` tunes the DLU path: `partition` (`"atoms"`, or `"single"` — one fragment,
which *is* the exact transformation through the same code and is the only like-for-like
reference for a DLU error) and `source` (`"diagonal"`, the default, using the molecular
diagonal blocks; or `"isolated"`, using separate isolated-atom problems, which is transferable
across a geometry scan).

### `screening` — the two-electron picture change

| value | what the Hamiltonian contains | cost |
|---|---|---|
| `"x2camf"` *(default)* | One-electron X2C **plus** the atomic mean-field two-electron picture change (both its spin-free and spin–orbit parts). | one four-component atomic SCF per unique element, cached |
| `"none"` | The one-electron X2C operator alone. Atomic j-splittings come out **5–30 % too large**. | free |
| `"x2camf-external"` | The same correction from the original authors' `x2camf` plugin. A bisection tool, never a default; it always uses the neutral atom as its reference and cannot decouple in a contracted basis. | as `"x2camf"` |
| `"mmf"` | ⚠ **Experimental benchmark only.** The same subtraction taken from a **full four-component SCF on the whole molecule** — no atomic approximation. Warns when selected. | a molecular 4c SCF with `(SS\|SS)`, growing as the fourth power of the basis |

⚠ **`"x2camf"` and `"mmf"` are values of the same option and cannot be combined** — they are
the same picture change, atomically and molecularly, so applying both would double-count it. On
an isolated closed-shell atom they solve the identical four-component problem; on a molecule
they differ by exactly the atom-diagonal approximation X2CAMF makes. Use `"mmf"` to measure
that difference on a small system, never to run a production calculation: its cost grows as the
fourth power of the basis, so anything with a heavy element in it is out of reach. That is why
X2CAMF exists.

On a finished stage, `scf.data.soc.provenance()` returns all of this as JSON — the method name, the decoupling
record (including `max |X|` per atom, the conditioning diagnostic) and the screening record. It
is what a stored property matrix must carry: a matrix that does not say which Hamiltonian
produced it is not interpretable, and the difference between the first two screening rows is
5–30 % on every splitting in it.

⚠ **`SpinOrbitX2C.screening` says what the Hamiltonian already CONTAINS — it is not a
request.** Anything adding a correction to a Hamiltonian whose record is not `"none"`
double-counts it; that is why the correction is applied in exactly one place, in the AO basis,
before any change of basis.

⚠ **Kuiva's exact decoupling and PySCF's are not the same to machine precision.** `"1e"` uses
PySCF's; the DLU code path uses Kuiva's, which additionally projects linear dependence out of
the four-component metric. They agree to ~1e-13 on light molecules and to ~2e-7 on a heavy one.
So comparing `"1e"` against `"1e-dlu"` measures the DLU approximation *plus* that difference —
use `decoupling_options={"partition": "single"}` for the like-for-like reference.

### ⚠ If you are comparing against another program

Everything below is a real difference that will show up as an apparent "method error" if it is
not matched. They are listed because they are invisible in the output otherwise.

- **Point nucleus.** Kuiva inherits PySCF's default. DIRAC defaults to a **Gaussian** nuclear
  charge distribution, and the difference is a genuine physical effect that **grows with Z**.
- **Speed of light.** `c = 137.03599967994` a.u., PySCF's value. Other codes ship other
  determinations; the difference is in the last digits but is not zero.
- **The decoupling is done in the decontracted basis** and the result contracted back, which is
  also what PySCF's X2C helper does. A code that decouples in the contracted basis gets a
  different (worse) operator.
- **Linear dependence is projected out of the four-component metric** at `1e-7` before the
  decoupling. PySCF removes none. The projected operator is the better-conditioned one.
- **The two decouplings are not the same `X`.** The one-electron part uses the exact molecular
  decoupling; the `"x2camf"` correction is atomic by construction, from the converged
  four-component atomic Fock. This is standard for X2CAMF, is recorded in the provenance, and
  `x2c_approx="atom1e"` makes the two consistent if you want to measure the difference.
- **Total energies do not contain the mean-field double-counting term.** Kuiva never reports
  one that does, so an absolute total is not directly comparable with a four-component total
  until that term has been accounted for. Relative energies — which is what everything here is
  about — are unaffected.
- ⚠ **Two rules bind any spin–orbit splitting quoted anywhere.** State **which construction
  produced it**: a self-consistent two-component calculation and a frozen-orbital
  diagonalization of the same operator differ by tens of per cent. And **never compare against
  a reference in a different basis** — no picture-change correction can or should recover a
  basis truncation, and in a small basis the two errors partly cancel, so a strictly better
  Hamiltonian can move a total *away* from experiment.

---

## Spin–orbit coupling

The ingested Hamiltonian carries the **two-electron picture change by default** (atomic
mean-field X2C), which is the difference between quantitative fine structure and j-splittings
5–30 % too large.

```python
scf = kuiva.ScalarSCF(mol)                                  # screening="x2camf" by default
scf = kuiva.ScalarSCF(mol, screening="none")                # one-electron X2C only
scf = kuiva.ScalarSCF(mol, screening_options={"interaction": "gaunt"})

scf.run().data.soc.screening.report()   # what the Hamiltonian CONTAINS, not what was asked for
```

⚠ The default costs **one four-component atomic solve per unique element** — under a second for
a light atom, tens of minutes for a lanthanide. It depends on no geometry, so it is paid once
ever per `(element, basis, charge, configuration, interaction)` and cached both in the process
and on disk (`~/.cache/kuiva/amf`, `$KUIVA_AMF_CACHE`, or `off`): a potential-energy surface
pays it once, not once per point. `screening="none"` is the escape hatch and the right choice
for anything that is not about spin–orbit coupling, where the correction is pure cost — it
changes no scalar quantity.

The reference configuration defaults to the neutral atom, except the f block, which defaults to
M(3+), on chemistry. Any oxidation state is overridable, and configurations across a molecule
are a **mapping**: a scalar `configuration` on a heteronuclear molecule raises, because "+3"
almost always means "the metal is trivalent". Open shells are occupied by average of
configuration, not aufbau.

The correction is also usable on its own, one element at a time:

```python
from pyscf import gto
from kuiva.amf import amf_correction

atom = gto.M(atom="Ne 0 0 0", basis="x2c-SVPall-2c")
corr = amf_correction(atom, method="x2camf")     # (Delta h_sf, Delta w) in the AO basis
corr.report()                                    # method, interaction, backend, magnitudes
```

---

## Choosing and inspecting an active space

Choosing an active space means answering "are these the orbitals I meant?" — and a **spinor
cannot simply be plotted**, so Kuiva gives you two ways to answer it. Both report **degenerate
blocks, not individual spinors**, by default: a single spinor's density and populations are
basis-dependent inside a degenerate manifold, and block sums are not.

### Löwdin populations: what an orbital is made of, as a number

```python
atomic, orbital = ref.population_analysis(level="active", active=active_columns)
atomic.report(spin_vector=True)          # charges and spin; |s| only by default
orbital.report(tolerance=0.01)           # reduced AO populations, contributions above 1 %
```

`level` chooses what gets a reduced-AO table — `"active"` (the default), `"frontier"` (a window
around the HOMO–LUMO gap) or `"all"` — and `tolerance` sets the smallest contribution printed.
`"all"` on a heavy element is thousands of rows, so it is never a default and warns when asked
for.

⚠ **Two things not to misread.** The **spin density of a state-averaged Kramers pair is exactly
zero** everywhere — that is time-reversal symmetry, not a lost moment; look at a single state
if the spin distribution is the question. And the **atomic charges are a basis-dependent
partition** that can disagree with a Mulliken charge on the same molecule not only in size but
in sign. Use charges to compare like with like, never as an oxidation state. The reduced
*orbital* populations are the robust half and are what this is for.

### Molden files: what is actually in them

```python
ref.write_molden("active.molden", columns=active_columns,
                 occupation=occ, energy=energies)      # one entry per Kramers pair
```

⚠ **A Kuiva molden file does not contain orbitals. It contains spinor densities, decomposed
into real components.** This is worth understanding before opening one.

A molden file can only hold real orbitals: a viewer evaluates ψ(r) = Σ c_μ χ_μ(r) and draws
isosurfaces of it. A two-component spinor is complex and has two components, and the square
root of its density ρ = |ψ^α|² + |ψ^β|² **is not expandable in the basis** — the standard case
is any m_j eigenfunction of a d shell, whose density is a torus that no real orbital squares
to. The tempting per-coefficient square root is badly wrong and produces an entirely plausible
picture.

So what Kuiva writes is the **exact** decomposition

&nbsp;&nbsp;&nbsp;&nbsp;ρ(r) = Σ_k w_k (u_k · χ(r))²,&nbsp;&nbsp; at most four components per Kramers pair,

with each `u_k` written as one molden orbital and `w_k` as its occupation. Consequently:

- **Each entry is one component of a density, not a spinor.** `Sym=` labels it
  `<spinors>_c<component>`, and the file's own header says so in plain text.
- **The number of components is the diagnostic.** One (weight 1.0) means the spinor really is a
  real orbital and the picture is the whole story. Two or more means it is not, and no single
  isosurface represents it — look at all of them.
- **Phases and signs are meaningless.** Only the squares entered the density.
- **Occupations are exact.** Summing `Occup × component²` over the file reproduces the electron
  density, and the occupations sum to the number of electrons. Empty orbitals are written too,
  with `Occup= 0`, since those are half of what you look at when choosing an active space.
- **Kramers partners have identical densities**, so one entry per pair is complete, not a
  simplification.

⚠ **h functions (l = 5) are outside the molden standard**, which defines up to `[9g]`. They are
**dropped by default**, with a warning and with the discarded weight measured and recorded in
the file header — a silently truncated orbital is a picture of something else.
`include_high_l=True` writes them anyway, continuing molden's own ordering with an `[11h]`
marker. That is **not standard**: many viewers will refuse or misread such a file, and it
exists for readers that do support it (e.g. [Kaijo](https://github.com/mansikkamaki/kaijo)).

---

## Configuration

### Files

Site defaults live in a `defaults.conf`, read from the first of these that exists:

| path | for |
|---|---|
| `$KUIVA_CONFIG` | an explicit file, for batch scripts |
| `~/.config/kuiva/defaults.conf` | per user — this is the one `setup.sh` writes |
| `/etc/kuiva/defaults.conf` | per machine (administrator) |
| `<sys.prefix>/etc/kuiva/defaults.conf` | per installation (administrator) |
| `<source tree>/defaults.conf` | a source checkout |

The release deliberately ships **none of them**: a memory limit must be chosen, not inherited
silently, which is why the first `source setup.sh` asks for one. Sections are for the reader's
benefit and are flattened when read; a malformed file is a hard error rather than a silent
fallback.

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
# scratch_dir = /scratch/$USER
# scratch_gb = 100.0
```

### Environment

| variable | meaning |
|---|---|
| `KUIVA_MEMORY_GB` | the memory limit for this job; overrides `defaults.conf` |
| `KUIVA_CONFIG` | an explicit configuration file |
| `KUIVA_NUM_THREADS` | the **total** number of threads the calculation may use |
| `KUIVA_KERNELS` | `auto` (default), `numpy`, or `native` |
| `KUIVA_AMF_CACHE` | directory for the atomic mean-field cache, or `off` |
| `KUIVA_SCRATCH`, `KUIVA_SCRATCH_GB` | scratch directory and the space Kuiva may use there |
| `KUIVA_CHECKPOINT_GB` | largest checkpoint file |

Anything passed explicitly to a call overrides the environment, which overrides the
configuration file.

### Threads: one number

```bash
export KUIVA_NUM_THREADS=4        # the total this calculation may use
```

That is the whole knob. Kuiva then spends it where the stage needs it — on the BLAS in a dense
stage such as the integral transform, and on the compiled kernels in a tensor-network sweep,
where a threaded BLAS measurably buys nothing and is charged for the spin-wait anyway. Run the
network solver at a small thread width and read its CPU-second figures accordingly.
`OMP_NUM_THREADS` is used when the knob is unset; with neither, Kuiva takes the cores it is
allowed to run on.

At startup Kuiva **measures** whether the BLAS in the process actually threads at the width you
asked for, and prints the verdict in the banner:

```
             threads: 4 (KUIVA_NUM_THREADS); BLAS: MKL 2025.1, threaded
```

It warns if the answer is no — a budget that buys no threads spends CPU-hours for nothing — and
it warns in the other direction too, when a second threading runtime is found spinning next to
the BLAS, because that silently inflates every timing the run reports. A BLAS with no thread
control at all is reported as "unverified" rather than as either verdict. To see the numbers
behind it:

```bash
python -m kuiva.util.threads
```

### Kernel backend

| `KUIVA_KERNELS` | meaning |
|---|---|
| `auto` (default) | compiled where available, NumPy otherwise; a missing build is not an error |
| `numpy` | pure NumPy even if the build exists — the reproducibility setting |
| `native` | refuse to start if the build is missing or stale (an explicit request is never quietly downgraded) |

---

## What a run writes

**The output file.** `add_file_handler(path)` mirrors the run's output stream to a file. It is
the human-readable output and nothing else: machine-readable matrices never enter it.

**The property matrices** (`PropertyDump`) — the product. A plain-text, line-oriented file with
`#` comments, a versioned `[HEADER]`, a `[PROVENANCE]` block of JSON, and one `i j Re Im`
record per matrix element: the effective Hamiltonian `H` and the three magnetic-moment
components `mu_x`, `mu_y`, `mu_z` in the basis of the spin–orbit eigenstates, in µ_B. It is
deliberately dull, because it is a contract with an external ITO / crystal-field code, and
`kuiva.props.dump.read_dump` is a working parser of it.

- ⚠ **`H` is diagonal**, unlike OpenMolcas RASSI's: this CI is already two-component, so its
  roots *are* the spin–orbit eigenstates and there is no separate spin–orbit mixing step. The
  header says so, because a reader coming from a two-step workflow will expect otherwise.
- ⚠ **Phases are arbitrary and are not canonicalized.** Within a degenerate block the
  eigenvectors are defined only up to a unitary mixing, so an element-by-element comparison of
  these matrices — against another program or against another run of this one — is meaningless.
  Compare through the invariants of `kuiva.props.multiplet`: degeneracy patterns, relative
  energies, and `M_ij = Tr_block(mu_i mu_j)` with its principal g values.
  `PropertyMatrices.analyse()` is that reduction, one call away, and free ions then have
  *analytic* targets (Landé g) independent of every program involved.
- ⚠ **No picture change is applied to `L` and `S`** — see [Limitations](#limitations--what-not-to-trust).
- The header carries the full Hamiltonian provenance, the gauge origin, and the active space as
  a physical statement rather than an index window.
- The **inactive contribution is computed and checked, never assumed away**: a Kramers-paired
  inactive set contributes exactly zero, and a nonzero result is a statement about the orbitals
  and warns.

**The pseudospin export** (`PseudospinExport`) — the multi-site sibling, for the external
OuluSpin code: the same dull shape, holding the effective Hamiltonian and moment operators on
the local-multiplet model space, the ordered pseudospin product basis, and the unitary mapping
the ab initio states onto it. ⚠ **Here `H` is *not* diagonal**, and the header says so;
`[ENERGIES]` lists its eigenvalues and `[MATRIX U]` the diagonalizing unitary. The `M`
convention and the storage order are OuluSpin's, restated in every file, so the file needs no
permutation on the way in. Spin operator matrices are deliberately not written.

Both files carry a `format_version` that is bumped when the *meaning* of a stored field
changes, so a consumer can refuse rather than misinterpret.

**Checkpoints** (`CASSCF(..., checkpoint=path)`) — schema-versioned HDF5, written every
macro-iteration under an adaptive budget, with the converged one always written. They hold the
orbitals and orbital-rotation state, the active-space RDMs, the state energies and the run
metadata; CI vectors only below a size threshold, as a Davidson warm start. Four-index
integrals are never stored — a restart regenerates them, which is cheap because the expensive
atomic solves have their own persistent cache.

⚠ **Failure semantics are the opposite of a cache's, on purpose.** A checkpoint *write* failure
is a warning and the run continues; a *read* failure on an explicitly requested restart is an
error that propagates, because silently starting over wastes exactly the hours the file existed
to protect. A schema mismatch refuses; a changed code fingerprint only warns.

---

## Examples

`examples/` holds complete, runnable calculations — one directory per pipeline stage, each a
commented script plus the committed output of a known-good run. The script's header comment is
the explanation; there is nothing else to read.

```bash
source setup.sh
cd examples/01_scf_and_reference
python scf_and_reference.py                    # writes output/scf_and_reference.out
diff output/scf_and_reference.out reference/scf_and_reference.ref.out
```

Every example asserts what it claims and exits non-zero if a check fails, so each one doubles
as a smoke test. In that `diff`, the timings, the memory summary and the banner's **version**
and **thread** lines are expected to differ; the physics is not. The references are generated
on the pure-NumPy path, so their banner reads `compiled kernels: disabled
(KUIVA_KERNELS=numpy)`; your run prints nothing there (the default, with no compiled backend
built) or `compiled kernels: native ...` (with one), and in the latter case quantities at the
1e-13 level may differ in their last digits. Also expected.

| # | example | what it shows | runtime |
|---|---|---|---|
| 1 | `01_scf_and_reference` | the front end end to end: scalar X2C SCF → orthonormal working basis → Kramers-paired spinors → factorized integrals, each stage checked | ~1 s |
| 2 | `02_atomic_spin_orbit` | fine structure of a free atom: the exact 2 + 4 splitting of a p shell, Landé g factors with no free parameter, and what two-electron screening changes | ~2 s |
| 3 | `03_casscf_ligand_field` | the flagship calculation: a state-averaged two-component CASSCF on TiCl₃, its active space stated as orbital character, and the five Kramers doublets of the d¹ ligand field | ~2 min |
| 4 | `04_dmrg_casscf` | the tensor-network solver: a cheap CI, a tree topology built from its entanglement, and a DMRG-CASSCF reproducing the exact CI through the same orbital optimizer | ~4 min |
| 5 | `05_nevpt2` | dynamic correlation: SC-NEVPT2 on the oxygen atom, its eight-class decomposition, term energies moving towards experiment while the degeneracies survive | ~3 s |
| 6 | `06_property_export` | the two products: the property-matrix dump and the OuluSpin pseudospin export, reaching the same g values by two independent routes | ~4 min |
| 7 | `07_checkpoint_restart` | a CASSCF checkpointed every macro-iteration, interrupted, and resumed from disk to the same energy | ~3 min |
| 8 | `08_slater_condon` | the extras: Slater–Condon parameters `F^k`, `G^k`, `R^k` and spin–orbit constants `ζ` of a free scandium atom, from an average-of-configuration reference | ~1 min |

## Extras

`kuiva.extras` holds self-contained special-purpose methods that ship with the code and are
usable, but are **not** part of the multireference pipeline above and are not maintained at the
same level. They reach the core through its ordinary public interfaces, and nothing in a
calculation depends on them.

There is currently one: **atomic Slater–Condon parameters**. Given an element and a
configuration you state, it converges an average-of-configuration scalar X2C SCF — spherical, one
radial function per shell — and returns the radial parameters `F^k(a,b)`, `G^k(a,b)` and
`R^k(ab;cd)` among the shells you name, together with the one-electron spin–orbit constants
`ζ_nl`, in a log table and in a versioned plain-text file.

```python
from kuiva.extras import slater_condon_parameters

result = slater_condon_parameters("Dy", "[Xe] 4f9 5d1 6s1", basis="x2c-TZVPall-2c",
                                  shells=("4f", "5d", "6s"), file="dy_i.scp")
```

⚠ The genuine cross parameters `R^k(ab;cd)` — the ones that are neither an `F^k` nor a `G^k` —
carry the **phase** of the radial functions they name an odd number of times, so their sign is
only meaningful against a stated convention. `kuiva` fixes each radial function positive in its
outer region (`P_nl(r) > 0` as `r → ∞`) and states this in every file it writes; `F^k`, `G^k`
and `ζ` are quadratic in every radial function they involve and never depended on it. Files
written before `format_version` 2 carry `R^k` signs that came from the eigensolver and may not
be compared across calculations.

⚠ The parameters are **frozen average-of-configuration** values of one fixed configuration in
one basis set: they are not self-consistent values for any particular term, they contain no
correlation, and they may not be compared against a value obtained in a different basis. A
parameter set fitted to experiment is a different object and Hartree–Fock-level values are known
to sit above one. Every file the feature writes states this, along with the `R^k` ordering
convention and the record saying whether the two-electron screening is inside `ζ`. `ζ=True`
(the default) needs the two-component operator and therefore a four-component atomic solve per
element — sub-second for a light atom, tens of minutes for a lanthanide, paid once and cached;
`zeta=False` keeps the run to the SCF. Example 8 is the worked demonstration.

## Tests

The default suite is laptop-fast and needs nothing beyond Kuiva's own dependencies.

```bash
source setup.sh
pip install '.[test]'
pytest                     # the default suite: fast tests only, budgeted at ~8 minutes
pytest -m slow             # the slow tests (heavy-atom four-component solves; hours)
pytest -m ''               # everything — budget several hours and pipe through `tee`
```

`tests/README.md` documents the tiers, the systems, the tolerance policy and how the committed
reference data was generated. The external programs (OpenMolcas, DIRAC) are needed only to
*regenerate* that data, which is committed; the suite as run compares against the stored files
and never re-runs them.

---

## Limitations — what not to trust

The release is usable for production work **with care**, and this is what the care is about.

- **No picture-change correction is applied to the property operators by default, and the
  approximation has been measured.** `L` and `S` are the bare non-relativistic AO operators,
  used unchanged in the two-component basis — what OpenMolcas RASSI does, which is what makes a
  cross-code comparison like-for-like. Every property dump warns and records which operators it
  used.

  The correction is available as a **non-default option**, `property_picture_change=True` on
  `scalar_x2c_reference` / `spinor_reference`. What it is worth, measured on nine systems:

  | | g shift, relative |
  |---|---|
  | `np¹` free ions, B (Z=5) → Tl (Z=81) | 1.0e-04 → 1.0e-03, smooth and monotone in Z |
  | Ce(3+) 4f¹ / Yb(3+) 4f¹³ | 1.3e-03 / 2.6e-03 |
  | Dy(3+) 4f⁹ ground multiplet | 5.1e-04 |
  | TiCl₃ ground doublet | 1.9e-04 |
  | any degeneracy, free ion or complex | **exactly zero** |

  So it is a real, orderly effect that is an order of magnitude below the 1% band the test
  suite asserts free-ion Landé factors at, which is why it is not the default. ⚠ **It grows
  with Z**, so those figures bound the elements measured and not a heavier one. ⚠ On a level
  whose `g` approaches zero the relative shift is inflated by its own denominator (the largest
  seen, 8e-03, is such a case) — quote an absolute shift too when `g` is small. ⚠ Turning it on
  changes what `mu` means in a stored file while `format_version` stays the same: the header
  field `picture_change_on_properties` is what distinguishes them, so reading the header is
  obligatory. The `g_e - 2` anomaly's own small-component term is a further three orders down
  and is left off unless `anomaly_picture_change=True`.
- ⚠ **DLU accuracy is unmeasured.** No statement about state energies or splittings through a
  `-DLU` Hamiltonian exists, so nothing computed with one may be quoted as a spectroscopic
  accuracy. It is the bottom rung of the cost ladder and warns when selected.
- ⚠ **Löwdin charges are the weakest number the code produces** and can come out with the
  opposite sign to a Mulliken charge on the same molecule. Reduced *orbital* populations are
  robust and are what the feature is for; a charge is never an oxidation state.
- **The conventional CI ceiling is about 20 spinors at half filling.** It is a memory bound on
  the *determinant count*, not on the spinor count, so it moves with the memory limit and is
  enforced before the first allocation — dilute or nearly-full spaces well past 20 spinors run
  comfortably. The hard limit is 64 spinors (a single 64-bit occupation mask). Beyond the
  ceiling, the tensor-network solver takes over.
- **The integral factorization is memory-bound.** The default Cholesky route currently
  materializes the conventional two-electron integral array, which grows as the fourth power of
  the basis and is reserved against the memory limit before the SCF. This, rather than core
  count, is what bounds the size of system that fits.
- **Kramers degeneracy in the general two-component CI emerges numerically, not by
  construction.** A Kramers-restricted (time-reversal-adapted) path is not implemented. In
  practice the measured splitting is far below the 1e-8–1e-6 Eh band reserved for a genuine
  numerical splitting, but it is not zero by symmetry. The **tensor-network** solver's figure is
  larger and for a different reason: a sweep converges its roots separately, leaving of order
  1e-9 Eh between paired states even where nothing is truncated. Both are far below anything
  physical; neither is a bound on the other.
- **No point-group or double-group symmetry is used.** States are the lowest *n* roots; there
  is no irrep selection, and a state average must be checked against the boundary diagnostic
  rather than assumed safe. Non-abelian symmetry adaptation is out of scope.
- **NEVPT2 is strongly contracted only, on a conventional-CI reference.** FIC and
  quasi-degenerate variants are not implemented and are not planned — the artificial multiplet
  splitting they would cure is four orders of magnitude below the size at which a splitting
  means different physics. A tensor-network reference has no NEVPT2 route: it would need a
  network-backed contraction provider, which is not built.
- **The tensor-network state is not checkpointed** through the class API, so a DMRG-CASSCF
  cannot be interrupted and resumed the way a conventional-CI one can.
- **One node, shared memory.** There is no MPI and no distributed tensor layer. Memory, not
  core count, is the scaling limit.
- **Magnetic properties themselves are out of scope.** Kuiva writes the operator matrices; the
  ITO / Stevens / crystal-field decomposition is an external code's job.
- **Four-component methods are out of scope.** Four-component machinery exists inside the code
  as an *ingredient* of the two-electron picture change, not as a method path.
- **An unrestricted reference gives spinors that are not Kramers paired**, so an active space
  cannot be taken as a contiguous spinor range there.
- **Absolute total energies** are not the quantity this code is tuned for: the Cholesky
  threshold default was decided on *relative* energies, and no usable threshold makes an
  absolute total accurate to 1e-8 Eh.

---

## Versioning

**Version 0.7.1.** The number is `MAJOR.MINOR.PATCH` and reads as usual:

| part | moves when |
|---|---|
| `MAJOR` | a **breaking change** — an API you have to change your script for, a stored-file layout an existing reader can no longer read, or a default whose change moves published numbers |
| `MINOR` | a **new capability**, with everything that worked before still working |
| `PATCH` | a **bug fix** or a minor adjustment that does not change intended behaviour |

While the version starts with `0`, the interface is not yet stable: `MAJOR` stays `0` and a
breaking change moves `MINOR` instead. Every commit carries a version bump, so a version
identifies exactly one state of the code — which is the point of printing it.

`kuiva.__version__` is the number, and it is also where every run and every product records
itself:

```python
import kuiva; kuiva.__version__          # '0.7.1'
```

- the run banner prints it, so the version is in the **output file**;
- the property dump and the pseudospin export carry it in their headers, as `code_version`;
- an HDF5 checkpoint stores it as the `kuiva_version` attribute;
- a molden file names it on the `written by kuiva` line.

Two other version numbers appear in those files and are **independent** of this one, because
they answer a different question — whether a file can be read at all: `format_version` in the
plain-text products and `schema_version` in a checkpoint. Both are bumped only when the
*meaning* or the *layout* of something already stored changes, so a new release usually leaves
them alone. Neither is what detects a numerically relevant code change: a checkpoint
additionally stores a source fingerprint, and a restart across a changed one warns.

A committed `examples/*/reference/*.ref.out` states the version it was generated under and is
not regenerated for a version bump.

---

## Citing Kuiva

`CITATION.cff` at the repository root is the machine-readable citation; GitHub renders it and
citation managers read it directly. Please also cite the methods, basis sets and libraries the
calculation actually used — they are listed in full below, and the `[PROVENANCE]` block of a
property dump names the Hamiltonian and basis you were actually running.

## Licence

Apache License 2.0; see `LICENSE` and `NOTICE`. The attribution obligation of the References
section is independent of it: every method, algorithm and basis set implemented here is someone
else's published work and is cited as such.

---

## References

The published methods, algorithms, basis sets and libraries that `kuiva` implements, follows or
depends on. Policy: **over-cite rather than under-cite**. Citations for the software used only to
generate validation reference data live with that code, in `tests/`.

### Libraries `kuiva` depends on

- **PySCF** (2.14.0) — scalar-relativistic X2C front-end (SCF guess and integrals); the only
  external dependency in the multireference path is its ingestion output. Q. Sun _et al._,
  _J. Chem. Phys._ **153**, 024109 (2020), DOI:10.1063/5.0006074; Q. Sun _et al._, _WIREs Comput.
  Mol. Sci._ **8**, e1340 (2018), DOI:10.1002/wcms.1340.
- **libcint** — Gaussian integral evaluation, used via PySCF and linked directly by `kuiva`'s
  compiled integral kernels. Q. Sun, _J. Comput. Chem._ **36**, 1664–1671 (2015),
  DOI:10.1002/jcc.23981.
- **HDF5** — checkpoint and cache format, via `h5py`. The HDF Group, Hierarchical Data Format v5,
  https://www.hdfgroup.org/HDF5/.
- **pybind11** (optional; only for the compiled kernel backend) — the C++/Python boundary of
  `cpp/kuiva_native.cpp`. W. Jakob, J. Rhinelander, D. Moldovan, _pybind11 — Seamless operability
  between C++11 and Python_ (2017), https://github.com/pybind/pybind11.
- **Intel oneAPI Math Kernel Library (MKL)** — the threaded BLAS/LAPACK behind NumPy on the
  reference build, called directly by the compiled kernels (`cblas_zgemm`) and asked for its
  per-region thread width (`MKL_Set_Num_Threads_Local`). Intel Corporation, oneAPI MKL Developer
  Reference, https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/.
- **OpenMP** (via Intel's `libiomp5`, optional; only for the compiled kernel backend) — the
  thread-level parallelism inside those kernels, including the composability rules that let MKL
  and a kernel share one runtime. OpenMP Architecture Review Board, _OpenMP Application
  Programming Interface_ 5.0 (2018), https://www.openmp.org/specifications/.
- **Basis Set Exchange** — authoritative basis data for the families PySCF does not bundle, used by
  `kuiva.basis.registry`. B. P. Pritchard, D. Altarawy, B. Didier, T. D. Gibson, T. L. Windus,
  _J. Chem. Inf. Model._ **59**, 4814–4820 (2019), DOI:10.1021/acs.jcim.9b00725; D. Feller,
  _J. Comput. Chem._ **17**, 1571 (1996); K. L. Schuchardt _et al._, _J. Chem. Inf. Model._ **47**,
  1045 (2007).

### Relativistic Hamiltonian (X2C)

- **Scalar / one-electron X2C (`sfx2c1e`)**, used for the front-end SCF guess. W. Liu, D. Peng,
  _J. Chem. Phys._ **131**, 031104 (2009), DOI:10.1063/1.3159445; D. Peng, M. Reiher, _Theor. Chem.
  Acc._ **131**, 1081 (2012), DOI:10.1007/s00214-011-1081-y; T. Nakajima, K. Hirao, _Chem. Rev._
  **112**, 385 (2012), DOI:10.1021/cr200040s.
- **Two-component X2C and exact decoupling**, the source of the ingested spin–orbit operator.
  W. Kutzelnigg, W. Liu, _J. Chem. Phys._ **123**, 241102 (2005), DOI:10.1063/1.2137315; W. Liu,
  D. Peng, _J. Chem. Phys._ **125**, 044102 (2006), DOI:10.1063/1.2222365; M. Iliaš, T. Saue,
  _J. Chem. Phys._ **126**, 064102 (2007), DOI:10.1063/1.2436882; D. Peng, M. Reiher, _Theor. Chem.
  Acc._ **131**, 1081 (2012), DOI:10.1007/s00214-011-1081-y.
- **Two-electron spin–orbit coupling by atomic mean-field X2C (X2CAMF)** — the default. J. Liu,
  L. Cheng, _J. Chem. Phys._ **148**, 144108 (2018), DOI:10.1063/1.5023750; and the atomic
  mean-field idea it builds on, B. A. Hess, C. M. Marian, U. Wahlgren, O. Gropen, _Chem. Phys.
  Lett._ **251**, 365 (1996), DOI:10.1016/0009-2614(96)00119-4 (AMFI). The reference implementation
  and its spherical-symmetry atomic solver: C. Zhang, L. Cheng, _J. Phys. Chem. A_ **126**, 4537
  (2022), DOI:10.1021/acs.jpca.2c02181.
- **Four-component Dirac–Hartree–Fock**, used inside `kuiva.amf` for the atomic reference.
  I. P. Grant, _Relativistic Quantum Theory of Atoms and Molecules_, Springer (2007);
  K. G. Dyall, K. Fægri, _Introduction to Relativistic Quantum Chemistry_, Oxford University Press
  (2007), ch. 4, 7 and 11 (Dirac–Coulomb, Gaunt and Breit operators). Average-of-configuration
  open-shell Dirac–Hartree–Fock: T. Saue, H. J. Aa. Jensen, _J. Chem. Phys._ **111**, 6211 (1999),
  DOI:10.1063/1.479958; J. Thyssen, PhD thesis, University of Southern Denmark (2001); the
  configuration-average energy functional itself, C. C. J. Roothaan, _Rev. Mod. Phys._ **32**, 179
  (1960), DOI:10.1103/RevModPhys.32.179.
- **Restricted kinetic balance**, the small-component normalization the four-component blocks and
  the X2C equations are written in. R. E. Stanton, S. Havriliak, _J. Chem. Phys._ **81**, 1910
  (1984), DOI:10.1063/1.447865; K. G. Dyall, K. Fægri, _Chem. Phys. Lett._ **174**, 25 (1990),
  DOI:10.1016/0009-2614(90)85321-3.
- **Picture change of properties and the renormalization matrix `R`**, which fixes how a
  four-component density maps onto a two-component one. D. Peng, M. Reiher, _J. Chem. Phys._ **136**,
  244108 (2012), DOI:10.1063/1.4729788.
- **Empirical spin–orbit screening — rejected, not deferred.** SNSO/Boettger factors and an
  in-house Breit–Pauli AMFI were both considered and rejected in favour of X2CAMF. Listed because
  the analysis followed them: J. C. Boettger, _Phys. Rev. B_ **57**, 8743 (1998),
  DOI:10.1103/PhysRevB.57.8743; M. Filatov, W. Zou, D. Cremer, _J. Chem. Phys._ **139**, 014106
  (2013), DOI:10.1063/1.4811776; B. de Souza, G. Farias, F. Neese, R. Izsák, _J. Chem. Theory
  Comput._ **15**, 1896 (2019), DOI:10.1021/acs.jctc.8b00841.

### Basis sets

- **Karlsruhe x2c-nZVPall / -2c** (segmented, H–Rn; the `-2c` recontraction is the default for
  two-component work). P. Pollak, F. Weigend, _J. Chem. Theory Comput._ **13**, 3696–3705 (2017),
  DOI:10.1021/acs.jctc.7b00593 (DZ/TZ); Y. J. Franzke, L. Spiske, P. Pollak, F. Weigend, _J. Chem.
  Theory Comput._ **16**, 5658–5674 (2020), DOI:10.1021/acs.jctc.0c00546 (QZ; and the `x2c-JFIT`
  Coulomb-fitting auxiliaries).
- **Peterson cc-pVnZ-X2C / cc-pwCVnZ-X2C** (mixed contraction; alkali/alkaline-earth, lanthanides,
  actinides; CBS-extrapolable). J. G. Hill, K. A. Peterson, _J. Chem. Phys._ **147**, 244106 (2018),
  DOI:10.1063/1.5010587 (K–Ra); Q. Lu, K. A. Peterson, _J. Chem. Phys._ **145**, 054111 (2016),
  DOI:10.1063/1.4959280 (La–Lu); R. Feng, K. A. Peterson, _J. Chem. Phys._ **147**, 084108 (2017),
  DOI:10.1063/1.4994725 (actinides); ccRepo, http://www.grant-hill.group.shef.ac.uk/ccrepo/.
- **Dyall basis sets** (uncontracted, heavy-element; benchmarking). K. G. Dyall, _Theor. Chem. Acc._
  **135**, 128 (2016), DOI:10.1007/s00214-016-1884-y, and the series of Dyall papers referenced
  therein.
- **ANO-RCC** (Douglas–Kroll–Hess-recontracted atomic natural orbitals, H–Cm). P.-O. Widmark,
  P.-Å. Malmqvist, B. O. Roos, _Theor. Chim. Acta_ **77**, 291 (1990), DOI:10.1007/BF01120130;
  B. O. Roos, R. Lindh, P.-Å. Malmqvist, V. Veryazov, P.-O. Widmark, _J. Phys. Chem. A_ **108**,
  2851 (2004), DOI:10.1021/jp031064+ (main group), **109**, 6575 (2005), DOI:10.1021/jp0581126
  (transition metals), and _Chem. Phys. Lett._ **409**, 295 (2005),
  DOI:10.1016/j.cplett.2005.05.011 (actinides).

### Orthogonalization and the working basis

- **Canonical and symmetric (Löwdin) orthogonalization.** P.-O. Löwdin, _J. Chem. Phys._ **18**, 365
  (1950), DOI:10.1063/1.1747632; P.-O. Löwdin, _Adv. Quantum Chem._ **5**, 185–199 (1970),
  DOI:10.1016/S0065-3276(08)60339-1. Textbook treatment: A. Szabo, N. S. Ostlund, _Modern Quantum
  Chemistry_, Dover (1996), sec. 3.4.5.
- **Linear-dependence removal by overlap-eigenvalue truncation.** J. Almlöf, K. Fægri, K. Korsell,
  _J. Comput. Chem._ **3**, 385 (1982), DOI:10.1002/jcc.540030314.
- **Cholesky orthogonalization** (the optional `cholesky` scheme). F. Aquilante, T. B. Pedersen,
  V. Veryazov, R. Lindh, _WIREs Comput. Mol. Sci._ **3**, 143 (2013), DOI:10.1002/wcms.1117;
  F. Aquilante, T. B. Pedersen, R. Lindh, _J. Chem. Phys._ **125**, 174101 (2006),
  DOI:10.1063/1.2360264.

### Spinor basis, time reversal, and Kramers pairs

- **Time reversal and Kramers degeneracy.** H. A. Kramers, _Proc. Amsterdam Acad._ **33**, 959
  (1930); E. P. Wigner, _Gruppentheorie und ihre Anwendung auf die Quantenmechanik der
  Atomspektren_, Vieweg (1931), ch. 26.
- **Kramers-paired spinor bases, the `−i σ_y K` convention, barred/unbarred notation.**
  K. G. Dyall, K. Fægri, _Introduction to Relativistic Quantum Chemistry_, Oxford University Press
  (2007), ch. 6 and 10; T. Saue, _ChemPhysChem_ **12**, 3077–3094 (2011),
  DOI:10.1002/cphc.201100682.
- **Time-reversal-adapted (Kramers-restricted) two-component MCSCF/CI.** J. Thyssen, T. Fleig,
  H. J. Aa. Jensen, _J. Chem. Phys._ **129**, 034109 (2008), DOI:10.1063/1.2943670; T. Fleig,
  J. Olsen, L. Visscher, _J. Chem. Phys._ **119**, 2963 (2003), DOI:10.1063/1.1590636.
- **Corresponding orbitals** (the standard way to pair two non-orthogonal orbital sets).
  A. T. Amos, G. G. Hall, _Proc. R. Soc. London A_ **263**, 483 (1961), DOI:10.1098/rspa.1961.0175.
- **UHF natural orbitals as an active-space guess.** P. Pulay, T. P. Hamilton, _J. Chem. Phys._
  **88**, 4926 (1988), DOI:10.1063/1.454704.

### Integral factorization and transformation

- **Cholesky decomposition of the two-electron integral matrix.** N. H. F. Beebe, J. Linderberg,
  _Int. J. Quantum Chem._ **12**, 683–705 (1977), DOI:10.1002/qua.560120408.
- **Pivoting, error bounds and modern practice.** H. Koch, A. Sánchez de Merás, T. B. Pedersen,
  _J. Chem. Phys._ **118**, 9481 (2003), DOI:10.1063/1.1578621; F. Aquilante, T. B. Pedersen,
  R. Lindh, _J. Chem. Phys._ **126**, 194106 (2007), DOI:10.1063/1.2736701; F. Aquilante _et al._,
  in _Linear-Scaling Techniques in Computational Chemistry and Physics_, Springer (2011),
  pp. 301–343, DOI:10.1007/978-90-481-2853-2_13.
- **Atomic / one-centre decomposition — the pivot selection used by default**, which keeps the
  factorization invariant under rotations of an atom instead of breaking degeneracies at the size
  of the threshold. F. Aquilante, R. Lindh, T. B. Pedersen, _J. Chem. Phys._ **127**, 114107
  (2007), DOI:10.1063/1.2777146.
- **Density fitting / resolution of the identity.** J. L. Whitten, _J. Chem. Phys._ **58**, 4496
  (1973), DOI:10.1063/1.1679012; B. I. Dunlap, J. W. D. Connolly, J. R. Sabin, _J. Chem. Phys._
  **71**, 3396 (1979), DOI:10.1063/1.438728; O. Vahtras, J. Almlöf, M. W. Feyereisen, _Chem. Phys.
  Lett._ **213**, 514 (1993), DOI:10.1016/0009-2614(93)89151-7.
- **Integral transformation by successive quarter transformations.** M. Yoshimine, IBM Technical
  Report RJ-555 (1969); T. Helgaker, P. Jørgensen, J. Olsen, _Molecular Electronic-Structure
  Theory_, Wiley (2000), ch. 9.
- **Two-component integral transformation** with spin-free AO integrals and complex spinor
  coefficients. L. Visscher, _Theor. Chem. Acc._ **98**, 68 (1997), DOI:10.1007/s002140050280;
  T. Saue, H. J. Aa. Jensen, _J. Chem. Phys._ **111**, 6211 (1999), DOI:10.1063/1.479958.

### Determinant CI, selected CI, and orbital optimization

- **Slater–Condon rules** and the phase conventions used for determinant matrix elements.
  J. C. Slater, _Phys. Rev._ **34**, 1293 (1929), DOI:10.1103/PhysRev.34.1293; E. U. Condon,
  _Phys. Rev._ **36**, 1121 (1930), DOI:10.1103/PhysRev.36.1121; T. Helgaker, P. Jørgensen,
  J. Olsen, _Molecular Electronic-Structure Theory_, Wiley (2000), ch. 1–2.
- **Bitmask determinant representation**, excitation analysis by XOR/popcount. A. Scemama,
  E. Giner, arXiv:1311.6244 (2013); Y. Garniron _et al._, "Quantum Package 2.0", _J. Chem. Theory
  Comput._ **15**, 3591 (2019), DOI:10.1021/acs.jctc.9b00176.
- **Perturbatively selected CI (CIPSI)** — the cheap CI's selection criterion. B. Huron,
  J. P. Malrieu, P. Rancurel, _J. Chem. Phys._ **58**, 5745 (1973), DOI:10.1063/1.1679199.
- **ASCI** — selection against a bounded set of generators, as implemented here. N. M. Tubman,
  J. Lee, T. Y. Takeshita, M. Head-Gordon, K. B. Whaley, _J. Chem. Phys._ **145**, 044112 (2016),
  DOI:10.1063/1.4955109.
- **Heat-bath selected CI.** A. A. Holmes, N. M. Tubman, C. J. Umrigar, _J. Chem. Theory Comput._
  **12**, 3674 (2016), DOI:10.1021/acs.jctc.6b00407.
- **CASSCF orbital gradients, the generalized Fock matrix and the exponential parametrization.**
  B. O. Roos, P. R. Taylor, P. E. M. Siegbahn, _Chem. Phys._ **48**, 157 (1980),
  DOI:10.1016/0301-0104(80)80045-0; P. E. M. Siegbahn, J. Almlöf, A. Heiberg, B. O. Roos, _J. Chem.
  Phys._ **74**, 2384 (1981), DOI:10.1063/1.441359; Helgaker, Jørgensen & Olsen (2000), ch. 10, 12.
- **Second-order / augmented-Hessian MCSCF.** H. J. Aa. Jensen, H. Ågren, _Chem. Phys. Lett._
  **110**, 140 (1984), DOI:10.1016/0009-2614(84)80166-1; H. J. Aa. Jensen, P. Jørgensen, _J. Chem.
  Phys._ **80**, 1204 (1984), DOI:10.1063/1.446797; D. A. Kreplin, P. J. Knowles, H.-J. Werner,
  _J. Chem. Phys._ **150**, 194106 (2019), DOI:10.1063/1.5094644.
- **One-index transformed Fock matrices**, which make a Hessian-vector product cost the same order
  as a gradient. P. Jørgensen, P. Swanstrøm, D. L. Yeager, _J. Chem. Phys._ **78**, 347 (1983),
  DOI:10.1063/1.444508; Helgaker, Jørgensen & Olsen (2000), ch. 10.8.
- **Inexact-Newton forcing sequences.** S. C. Eisenstat, H. F. Walker, _SIAM J. Sci. Comput._ **17**,
  16 (1996), DOI:10.1137/0917003; R. S. Dembo, S. C. Eisenstat, T. Steihaug, _SIAM J. Numer. Anal._
  **19**, 400 (1982), DOI:10.1137/0719025.
- **Trust-region methods and level shifting.** J. Nocedal, S. J. Wright, _Numerical Optimization_,
  2nd ed., Springer (2006), ch. 4; T. Helgaker, _Chem. Phys. Lett._ **182**, 503 (1991),
  DOI:10.1016/0009-2614(91)90115-P.
- **L-BFGS.** J. Nocedal, _Math. Comput._ **35**, 773 (1980),
  DOI:10.1090/S0025-5718-1980-0572855-7; D. C. Liu, J. Nocedal, _Math. Program._ **45**, 503 (1989),
  DOI:10.1007/BF01589116.
- **Two-component / relativistic MCSCF**, where the rotation parameters are complex.
  H. J. Aa. Jensen, K. G. Dyall, T. Saue, K. Fægri, _J. Chem. Phys._ **104**, 4083 (1996),
  DOI:10.1063/1.471644; T. Fleig, J. Olsen, C. M. Marian, _J. Chem. Phys._ **114**, 4775 (2001),
  DOI:10.1063/1.1349076.
- **Natural orbitals and their occupation numbers as an active-space criterion.** P.-O. Löwdin,
  _Phys. Rev._ **97**, 1474 (1955), DOI:10.1103/PhysRev.97.1474; P. Pulay, T. P. Hamilton,
  _J. Chem. Phys._ **88**, 4926 (1988), DOI:10.1063/1.454704.

### Orbital entanglement

- **Orbital entanglement entropies and mutual information.** J. Rissler, R. M. Noack, S. R. White,
  _Chem. Phys._ **323**, 519 (2006), DOI:10.1016/j.chemphys.2005.10.018; Ö. Legeza, J. Sólyom,
  _Phys. Rev. B_ **68**, 195116 (2003), DOI:10.1103/PhysRevB.68.195116.
- **Entanglement-based orbital ordering and the Fiedler vector.** G. Barcza, Ö. Legeza,
  K. H. Marti, M. Reiher, _Phys. Rev. A_ **83**, 012508 (2011), DOI:10.1103/PhysRevA.83.012508;
  M. Fiedler, _Czechoslovak Math. J._ **23**, 298 (1973).
- **Entanglement-driven automated active-space selection.** C. J. Stein, M. Reiher, _J. Chem.
  Theory Comput._ **12**, 1760 (2016), DOI:10.1021/acs.jctc.6b00156.
- **Review of the measures and conventions.** K. Boguslawski, P. Tecmer, _Int. J. Quantum Chem._
  **115**, 1289 (2015), DOI:10.1002/qua.24832.

### Multireference methods

- **The complete-active-space SCF method.** B. O. Roos, P. R. Taylor, P. E. M. Siegbahn, _Chem.
  Phys._ **48**, 157 (1980), DOI:10.1016/0301-0104(80)80045-0; the Newton–Raphson formulation
  the second-order orbital step follows: P. E. M. Siegbahn, J. Almlöf, A. Heiberg, B. O. Roos,
  _J. Chem. Phys._ **74**, 2384 (1981), DOI:10.1063/1.441359; H.-J. Werner, P. J. Knowles,
  _J. Chem. Phys._ **82**, 5053 (1985), DOI:10.1063/1.448627. Working equations and the
  redundant-rotation classification: T. Helgaker, P. Jørgensen, J. Olsen, _Molecular
  Electronic-Structure Theory_, Wiley (2000), ch. 10 and 12.
- **DMRG and tree tensor networks (the `kuiva.dmrg` solver).** Two-site DMRG: S. R. White,
  _Phys. Rev. Lett._ **69**, 2863 (1992), DOI:10.1103/PhysRevLett.69.2863; review of canonical
  forms, sweeps and truncation: U. Schollwöck, _Ann. Phys._ **326**, 96 (2011),
  DOI:10.1016/j.aop.2010.09.012. Ab initio complementary operators: T. Xiang, _Phys. Rev. B_
  **53**, R10445 (1996), DOI:10.1103/PhysRevB.53.R10445; S. R. White, R. L. Martin, _J. Chem.
  Phys._ **110**, 4127 (1999), DOI:10.1063/1.478522. MPO/TTNO compilation: C. Hubig,
  I. P. McCulloch, U. Schollwöck, _Phys. Rev. B_ **95**, 035129 (2017),
  DOI:10.1103/PhysRevB.95.035129; G. K.-L. Chan, A. Keselman, N. Nakatani, Z. Li, S. R. White,
  _J. Chem. Phys._ **145**, 014102 (2016), DOI:10.1063/1.4955108. Tree tensor network states:
  Y.-Y. Shi, L.-M. Duan, G. Vidal, _Phys. Rev. A_ **74**, 022320 (2006),
  DOI:10.1103/PhysRevA.74.022320; V. Murg, F. Verstraete, Ö. Legeza, R. M. Noack, _Phys. Rev. B_
  **82**, 205105 (2010), DOI:10.1103/PhysRevB.82.205105; N. Nakatani, G. K.-L. Chan, _J. Chem.
  Phys._ **138**, 134113 (2013), DOI:10.1063/1.4798639; K. Gunst, F. Verstraete, S. Wouters,
  Ö. Legeza, D. Van Neck, _J. Chem. Theory Comput._ **14**, 2026 (2018),
  DOI:10.1021/acs.jctc.8b00098. State-averaged DMRG in a shared basis: J. J. Dorando,
  J. Hachmann, G. K.-L. Chan, _J. Chem. Phys._ **127**, 084109 (2007), DOI:10.1063/1.2768360.
  Dynamic block selection: Ö. Legeza, J. Röder, B. A. Hess, _Phys. Rev. B_ **67**, 125114
  (2003), DOI:10.1103/PhysRevB.67.125114. Automatic structural optimization of tree tensor
  networks (the adaptive-topology moves): T. Hikihara, H. Ueda, K. Okunishi, K. Harada,
  T. Nishino, _Phys. Rev. Research_ **5**, 013031 (2023), DOI:10.1103/PhysRevResearch.5.013031.
  Symmetry-blocked tensors: S. Singh, R. N. C. Pfeifer, G. Vidal, _Phys. Rev. B_ **83**, 115125
  (2011), DOI:10.1103/PhysRevB.83.115125. Relativistic (general-spinor) DMRG: S. Knecht,
  Ö. Legeza, M. Reiher, _J. Chem. Phys._ **140**, 041101 (2014), DOI:10.1063/1.4862495;
  H. Zhai et al., _J. Chem. Phys._ **159**, 234801 (2023), DOI:10.1063/5.0180424.
  Jordan–Wigner transformation: P. Jordan, E. Wigner, _Z. Phys._ **47**, 631 (1928),
  DOI:10.1007/BF01331938. Higher-order reduced density matrices from matrix-product
  states for multireference perturbation theory (the cumulant-free route of the network
  3-/4-RDMs): Y. Kurashige, T. Yanai, _J. Chem. Phys._ **135**, 094104 (2011),
  DOI:10.1063/1.3629454; S. Guo, M. A. Watson, W. Hu, Q. Sun, G. K.-L. Chan, _J. Chem.
  Theory Comput._ **12**, 1583 (2016), DOI:10.1021/acs.jctc.6b00118.
- **Effective Hamiltonians on a local-multiplet model space** (the route from a converged
  network to the multi-site anisotropic-exchange operator, in place of solving thousands of
  individual roots). C. Bloch, _Nucl. Phys._ **6**, 329 (1958),
  DOI:10.1016/0029-5582(58)90116-0; J. des Cloizeaux, _Nucl. Phys._ **20**, 321 (1960),
  DOI:10.1016/0029-5582(60)90177-2. The Rayleigh–Ritz block compression used here is the
  zeroth CORE step: C. J. Morningstar, M. Weinstein, _Phys. Rev. D_ **54**, 4131 (1996),
  DOI:10.1103/PhysRevD.54.4131.
- **SC-NEVPT2 and the Dyall zeroth-order Hamiltonian.** C. Angeli, R. Cimiraglia, S. Evangelisti,
  T. Leininger, J.-P. Malrieu, _J. Chem. Phys._ **114**, 10252 (2001), DOI:10.1063/1.1361246;
  C. Angeli, R. Cimiraglia, J.-P. Malrieu, _J. Chem. Phys._ **117**, 9138 (2002),
  DOI:10.1063/1.1515317; K. G. Dyall, _J. Chem. Phys._ **102**, 4909 (1995), DOI:10.1063/1.469539.
  Review and the class-by-class working equations: C. Angeli, M. Pastore, R. Cimiraglia,
  _Theor. Chem. Acc._ **117**, 743 (2007), DOI:10.1007/s00214-006-0207-0. ⚠ Every one of those
  equations is spin-free and real; the ones `kuiva` implements were re-derived in spinor second
  quantization for a complex two-component Hamiltonian with 4-fold integral symmetry only, and
  the derivations are in the `kuiva/pt/` docstrings.
- **Avoiding the stored four-particle density matrix in NEVPT2.** K. Kollmar, K. Sivalingam,
  Y. Guo, F. Neese, _J. Chem. Phys._ **155**, 234104 (2021), DOI:10.1063/5.0072129; and the
  matrix-free precedent for full-CI solvers in PySCF's `mrpt.NEVPT`. `kuiva` goes one step
  further and contracts the integrals into one perturber vector per external label, so no
  rank-3 or rank-4 object is formed at all.
- **Level shifts for intruder states.** B. O. Roos, K. Andersson, _Chem. Phys. Lett._ **245**,
  215 (1995), DOI:10.1016/0009-2614(95)01010-7 (real); N. Forsberg, P.-Å. Malmqvist,
  _Chem. Phys. Lett._ **274**, 196 (1997), DOI:10.1016/S0009-2614(97)00669-6 (imaginary).
- **Quasi-degenerate NEVPT2 and model-space invariance** (not implemented and not planned;
  records the measurement that decided it). C. Angeli, S. Borini, M. Cestari, R. Cimiraglia, _J. Chem. Phys._ **121**, 4043
  (2004), DOI:10.1063/1.1778711; A. A. Granovsky, _J. Chem. Phys._ **134**, 214113 (2011),
  DOI:10.1063/1.3596699; S. Sharma, G. Jeanmairet, A. Alavi, _J. Chem. Phys._ **144**, 034103
  (2016), DOI:10.1063/1.4939752. Spin–orbit QD-NEVPT2, the closest prior art: R. Majumder,
  A. Yu. Sokolov, _J. Phys. Chem. A_ **127**, 546 (2023), DOI:10.1021/acs.jpca.2c07953.

### Quantum-computing CI solvers

- **Jordan–Wigner transformation** — the mapping from Kuiva's spinor occupation strings to
  qubits. P. Jordan, E. Wigner, _Z. Phys._ **47**, 631 (1928), DOI:10.1007/BF01331938. The
  symplectic (X-mask, Z-mask) bookkeeping and the standard statement of the electronic-structure
  mapping: J. T. Seeley, M. J. Richard, P. J. Love, _J. Chem. Phys._ **137**, 224109 (2012),
  DOI:10.1063/1.4768229. The alternative encodings the mapping registry stays open to:
  S. B. Bravyi, A. Y. Kitaev, _Ann. Phys._ **298**, 210 (2002), DOI:10.1006/aphy.2002.6254.
- **Sample-based quantum diagonalization** (the primary algorithm here — a quantum circuit as a
  *configuration sampler*, with the Hamiltonian diagonalized classically in the sampled
  subspace). "Chemistry beyond the scale of exact diagonalization on a quantum-centric
  supercomputer", _Sci. Adv._ (2025), DOI:10.1126/sciadv.adu9991 (full author list at the DOI);
  "Localized sample-based quantum diagonalization for strongly correlated chemistry", _PNAS_
  (2025), DOI:10.1073/pnas.2603914123. The sample-based **Krylov** variant, sampling Trotterized
  time-evolved states instead of an ansatz: "Sample-based Krylov quantum diagonalization",
  arXiv:2501.09702 (2025); the wider quantum-subspace family it belongs to, M. Motta et al.,
  _Electron. Struct._ **6**, 013001 (2024), DOI:10.1088/2516-1075/ad3592.
- **Unitary coupled cluster and cluster-Jastrow ansätze**, generalized here to complex spinor
  excitation generators. J. Romero, R. Babbush, J. R. McClean, C. Hempel, P. J. Love,
  A. Aspuru-Guzik, _Quantum Sci. Technol._ **4**, 014008 (2018), DOI:10.1088/2058-9565/aad3e4;
  F. A. Evangelista, G. K.-L. Chan, G. E. Scuseria, _J. Chem. Phys._ **151**, 244112 (2019),
  DOI:10.1063/1.5133059; A. Kandala et al., _Nature_ **549**, 242 (2017), DOI:10.1038/nature23879
  for the hardware-efficient family used as the structure-free control.
- **Fermionic circuit compilation** — Pauli-string exponentials and adjacent-mode Givens
  networks, which is how an orbital rotation is realized exactly rather than Trotterized.
  I. D. Kivlichan, J. McClean, N. Wiebe, C. Gidney, A. Aspuru-Guzik, G. K.-L. Chan, R. Babbush,
  _Phys. Rev. Lett._ **120**, 110501 (2018), DOI:10.1103/PhysRevLett.120.110501; the triangular
  decomposition itself, M. Reck, A. Zeilinger, H. J. Bernstein, P. Bertani, _Phys. Rev. Lett._
  **73**, 58 (1994), DOI:10.1103/PhysRevLett.73.58. Product formulas: H. F. Trotter,
  _Proc. Am. Math. Soc._ **10**, 545 (1959); M. Suzuki, _Commun. Math. Phys._ **51**, 183 (1976),
  DOI:10.1007/BF01609348.
- **Variational quantum eigensolver** (the secondary, exploratory path). A. Peruzzo et al.,
  _Nat. Commun._ **5**, 4213 (2014), DOI:10.1038/ncomms5213; J. R. McClean, J. Romero,
  R. Babbush, A. Aspuru-Guzik, _New J. Phys._ **18**, 023023 (2016),
  DOI:10.1088/1367-2630/18/2/023023. Its gradient is taken by the parameter-shift rule:
  K. Mitarai, M. Negoro, M. Kitagawa, K. Fujii, _Phys. Rev. A_ **98**, 032309 (2018),
  DOI:10.1103/PhysRevA.98.032309; M. Schuld, V. Bergholm, C. Gogolin, J. Izaac, N. Killoran,
  _Phys. Rev. A_ **99**, 032331 (2019), DOI:10.1103/PhysRevA.99.032331. The derivative-free
  alternative is COBYLA: M. J. D. Powell, in _Advances in Optimization and Numerical Analysis_
  (1994), DOI:10.1007/978-94-015-8330-5_4.
- **Qubit tapering from Z₂ Pauli symmetries** — cited for the *negative* result that it does not
  apply to time reversal, which is antiunitary and has no Pauli generator. S. Bravyi,
  J. M. Gambetta, A. Mezzacapo, K. Temme, arXiv:1701.08213 (2017).
- **Qiskit and Qiskit Aer** — the first backend adapter and the local simulator, testing-only and
  never a runtime dependency of `kuiva`. A. Javadi-Abhari et al., "Quantum computing with
  Qiskit", arXiv:2405.08810 (2024).

### Population analysis and orbital visualization

- **Löwdin population analysis.** P.-O. Löwdin, _J. Chem. Phys._ **18**, 365 (1950),
  DOI:10.1063/1.1747632; P.-O. Löwdin, _Adv. Quantum Chem._ **5**, 185–199 (1970),
  DOI:10.1016/S0065-3276(08)60339-1.
- **Basis-set dependence of Mulliken and Löwdin partitions** (why a charge from either is for
  comparing like with like, not a physical oxidation state). I. Mayer, _Chem. Phys. Lett._ **393**,
  209 (2004), DOI:10.1016/j.cplett.2004.06.031. Original partition: R. S. Mulliken,
  _J. Chem. Phys._ **23**, 1833 (1955), DOI:10.1063/1.1740588.
- **Natural-orbital decomposition of a one-particle density matrix** — what the spinor-density
  decomposition is, applied to a rank ≤ 4 density. P.-O. Löwdin, _Phys. Rev._ **97**, 1474 (1955),
  DOI:10.1103/PhysRev.97.1474.
- **The molden file format**, its AO ordering and normalization conventions. G. Schaftenaar,
  J. H. Noordik, _J. Comput.-Aided Mol. Design_ **14**, 123 (2000), DOI:10.1023/A:1008193805436;
  G. Schaftenaar, E. Vlieg, G. Vriend, _J. Comput.-Aided Mol. Design_ **31**, 789 (2017),
  DOI:10.1007/s10822-017-0042-5.
- **Real solid-harmonic Gaussian conventions.** H. B. Schlegel, M. J. Frisch,
  _Int. J. Quantum Chem._ **54**, 83 (1995), DOI:10.1002/qua.560540202.

### Spin–orbit multiplets, magnetic moments, and pseudospin

- **Pseudospin Hamiltonians and the ab initio g-tensor** — the `g gᵀ` construction used for the
  phase-invariant moment reductions. L. F. Chibotaru, L. Ungur, _J. Chem. Phys._ **137**, 064112
  (2012), DOI:10.1063/1.4739763.
- **Pseudospin / g-tensor conventions for Kramers doublets.** A. Abragam, B. Bleaney, _Electron
  Paramagnetic Resonance of Transition Ions_, Clarendon Press, Oxford (1970).
- **Landé g factors and free-ion multiplets.** R. D. Cowan, _The Theory of Atomic Structure and
  Spectra_, University of California Press (1981), ch. 11.

- **Magnetic-moment operators from spin–orbit eigenstates**, `μ = −(L + g_e S) μ_B`, evaluated
  through one-particle transition density matrices. J. Olsen, B. O. Roos, P. Jørgensen,
  H. J. Aa. Jensen, _J. Chem. Phys._ **89**, 2185 (1988), DOI:10.1063/1.455063. ⚠ `kuiva` applies
  **no picture-change transformation** to `L` and `S` (the same choice OpenMolcas RASSI makes);
  what removing it would require is D. Peng, M. Reiher, _J. Chem. Phys._ **136**, 244108 (2012),
  DOI:10.1063/1.4729788.
- **Free-electron g factor.** CODATA 2018 recommended values, E. Tiesinga, P. J. Mohr,
  D. B. Newell, B. N. Taylor, _Rev. Mod. Phys._ **93**, 025010 (2021),
  DOI:10.1103/RevModPhys.93.025010.

### Two-component CASSCF and the CI sigma vector

- **Two-component (spinor) CASSCF/CI**, where the CI roots are already the spin–orbit eigenstates
  and no separate spin–orbit mixing step exists. H. J. Aa. Jensen, K. G. Dyall, T. Saue, K. Fægri,
  _J. Chem. Phys._ **104**, 4083 (1996), DOI:10.1063/1.471644; T. Fleig, J. Olsen, C. M. Marian,
  _J. Chem. Phys._ **114**, 4775 (2001), DOI:10.1063/1.1349076.
- **Two-step `E_pq` resolution of the CI sigma vector** — the gather / dense GEMM / gather
  algorithm implemented here. B. O. Roos, _Chem. Phys. Lett._ **15**, 153 (1972),
  DOI:10.1016/0009-2614(72)80140-4; P. E. M. Siegbahn, _J. Chem. Phys._ **72**, 1647 (1980),
  DOI:10.1063/1.439365.
- **String-driven CI and the `h̃` one-electron folding.** P. J. Knowles, N. C. Handy, _Chem. Phys.
  Lett._ **111**, 315 (1984), DOI:10.1016/0009-2614(84)85513-X; J. Olsen, B. O. Roos, P. Jørgensen,
  H. J. Aa. Jensen, _J. Chem. Phys._ **89**, 2185 (1988), DOI:10.1063/1.455063.
- **Lexicographic combinatorial ranking of occupation strings** (the complete-CAS address map, in
  place of a hash table). D. E. Knuth, _The Art of Computer Programming_, Vol. 4A, sec. 7.2.1.3,
  Addison-Wesley (2011); T. Helgaker, P. Jørgensen, J. Olsen, _Molecular Electronic-Structure
  Theory_, Wiley (2000), ch. 11.

### Extras: atomic Slater–Condon parameters

- **The radial parameters `F^k`, `G^k`, `R^k` and the `c^k` coefficients**, in the ordering and
  phase conventions used throughout the feature. E. U. Condon, G. H. Shortley, _The Theory of
  Atomic Spectra_, Cambridge University Press (1935), ch. VI and XI; J. C. Slater, "The Theory
  of Complex Spectra", _Phys. Rev._ **34**, 1293 (1929), DOI:10.1103/PhysRev.34.1293.
- **Configuration-average energies, average-of-configuration practice, and the parameter
  conventions the output follows.** R. D. Cowan, _The Theory of Atomic Structure and Spectra_,
  University of California Press (1981), ch. 6, 10 and 14.
- **The 3j symbols**, evaluated in exact rational arithmetic from the single-sum formula.
  G. Racah, "Theory of Complex Spectra. II", _Phys. Rev._ **62**, 438 (1942),
  DOI:10.1103/PhysRev.62.438.
- **Hydrogenic spin–orbit constants**, the closed form the fit is validated against.
  H. A. Bethe, E. E. Salpeter, _Quantum Mechanics of One- and Two-Electron Atoms_, Springer
  (1957), sec. 12.
- The average-of-configuration mean field and the real solid-harmonic convention are cited above,
  under **Relativistic Hamiltonian (X2C)** and **Basis sets** respectively.

### Numerical methods

- **Davidson eigensolver** and its diagonal preconditioner. E. R. Davidson, _J. Comput. Phys._
  **17**, 87 (1975), DOI:10.1016/0021-9991(75)90065-0. The block (simultaneous-expansion)
  generalization implemented here: B. Liu, "The simultaneous expansion method", in _Numerical
  Algorithms in Chemistry: Algebraic Methods_, LBL-8158, Lawrence Berkeley Laboratory (1978),
  p. 49. Convergence analysis and the role of the restart subspace: M. Crouzeix, B. Philippe,
  M. Sadkane, _SIAM J. Sci. Comput._ **15**, 62 (1994), DOI:10.1137/0915004.
- **Re-orthogonalized modified Gram–Schmidt.** Å. Björck, _BIT_ **7**, 1 (1967),
  DOI:10.1007/BF01934122; G. H. Golub, C. F. Van Loan, _Matrix Computations_, 4th ed., Johns
  Hopkins (2013), sec. 5.2.
- **Sparse (list-of-transitions) operator storage and its contraction**, the form the compiled
  TTNO is kept in. C. Hubig, I. P. McCulloch, U. Schollwöck, _Phys. Rev. B_ **95**, 035129
  (2017), DOI:10.1103/PhysRevB.95.035129 (also cited above for the compilation itself).
- Higher-RDM contraction from the network is cited with the DMRG solver under **Multireference
  methods** (Kurashige & Yanai 2011; Guo _et al._ 2016).
