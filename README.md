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
- **The deliverable is a file**: the effective Hamiltonian, the three magnetic-moment operators
  and the three electric-dipole operators in the basis of the spin–orbit eigenstates, for an
  external crystal-field / ITO analysis code. Kuiva writes the operators and their invariants
  and does not do that analysis itself.

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

**Several calculations in one script** need one more line, because a reservation lives as long
as the process: nothing watches the arrays, so a finished calculation's memory stays on the
ledger until it is given back, and the *n*-th system in a loop is otherwise refused against a
limit its predecessors filled. Scope each one:

```python
from kuiva.util import resources

for system in systems:
    with resources.calculation(system.label):    # released at the end of the block
        run(system)
```

`resources.clear()` does the same thing for a driver whose structure does not suit a `with`
block. Neither weakens the check itself — a calculation is still refused before it allocates —
and if it does happen, the refusal says how much of the shortfall came from an earlier
calculation rather than letting it look like a machine that is too small.

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

   A clean gap is necessary and not sufficient, so the converged boundary report also states
   the **spin non-invariance** of the averaged density — whether the ensemble is one the
   symmetry leaves invariant at all. Near zero (every term-complete or full-space average)
   means safe by symmetry, and the gap is irrelevant. Large (any single-J or single-doublet
   average — measured 0.07–0.70, cleanly separated from the invariant class) means the
   average *leans* on the
   spin–orbit structure and is protected only by its boundary gap and its starting orbitals.
   Measured across free ions and ligand-field complexes: a leaning average over a ~30 cm⁻¹
   gap is a saddle — a 10⁻⁸-sized Kramers defect grows by orders of magnitude per iteration
   until the run is refused; leaning averages over gaps of a few hundred cm⁻¹ and up were
   locally stable, ligand-field ground-doublet averages among them. The case to know about
   is the free ion with several open-shell electrons: a J-only average that is locally
   stable at the symmetric orbitals still converged *from the scalar guess* to a solution
   with the `2J+1` manifold split by ~2 cm⁻¹, every diagnostic clean — a wrong basin, which
   no check can see from inside the run. Average whole terms; if a sub-manifold average is
   what you want, converge the whole-term (or full-space) average first, start the
   sub-manifold run from those orbitals, and check the manifold splitting of the result
   directly.

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
| `CASCI` *(optional)* | `Reference`, `CheapCI` or `CASSCF` | a full CI at **fixed** orbitals: a spectrum, a symmetry mode or an active space varied without a second orbital optimization | `.energy`, `.energies`, `.coeff`, `.result` |
| `NEVPT2` *(optional)* | `CASSCF` or `CASCI` | SC-NEVPT2, per state, by excitation class | `.e2`, `.total_energies`, `.class_energies` |
| `PropertyDump` | `CASSCF`, `CASCI` or `NEVPT2` | the property-matrix file: `H`, `mu_x`, `mu_y`, `mu_z` | `.matrices`, `.path` |
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
`memory_gb`, `gauge_origin`, `auxbasis`, `conv_tol`, `max_cycle`, and the convergence
controls below.

`reference="auto"` picks RHF for a closed shell and ROHF otherwise; RHF on an open shell is
refused rather than silently promoted.

#### When the SCF will not converge

This is where a real calculation first stops. An ROHF on a metal ion with several shells
within an eV of each other oscillates between occupations, and no amount of `max_cycle` fixes
it — so ⚠ **an SCF that runs out of cycles refuses**, rather than handing on whichever iterate
the budget stopped at. Everything downstream is built on those orbitals, and while the CASSCF
re-optimizes them, that is a hope and not a property. `allow_unconverged_scf=True` proceeds
deliberately and warns.

The levers, roughly in the order worth trying them:

| option | what it does |
|---|---|
| `level_shift=0.2` | adds an energy to the virtual orbitals, so the occupations stop swapping back and forth |
| `damp=0.5` | mixes the previous Fock matrix in; slower, steadier |
| `diis="adiis"` | the energy-based DIIS variant instead of Pulay's commutator one (`"cdiis"`, the default); much better in the first iterations of a hard open-shell case. `"ediis"` and `diis=False` also exist, with `diis_space=` and `diis_start_cycle=` |
| `init_guess="atom"` | a different starting density (`"minao"` default, plus `"atom"`, `"1e"`/`"hcore"`, `"huckel"`, `"mod_huckel"`, `"sap"`) |
| `second_order=True` | the CIAH Newton solver: converges cases the DIIS iteration cannot, at more cost per iteration and far fewer of them. ⚠ It does not use the four options above, and asking for both warns |

```python
scf = kuiva.ScalarSCF(complex_, level_shift=0.2, diis="adiis").run()
scf = kuiva.ScalarSCF(complex_, second_order=True).run()      # when that is not enough
```

⚠ **An unrecognized `init_guess` is refused, not substituted.** PySCF falls back to `minao`
silently for a name it does not know, which would run a different calculation from the one you
asked for.

#### Is the converged solution a minimum at all?

`stability="check"` runs the **internal** stability analysis on the converged SCF and reports
the verdict; `stability="follow"` also rotates into the unstable mode and re-solves, up to
three times. Off by default, because it costs a Davidson over the orbital Hessian.

It is worth that cost whenever the reference is in doubt. ⚠ **An unstable SCF is a saddle
point of the SCF energy** — a converged flag, a small gradient and a plausible energy are all
present, and a lower solution of the same reference exists one rotation away. Measured on a
Ni atom in `x2c-SVPall-2c`: the ROHF converges cleanly at −1518.501 Eh, and `stability=
"follow"` lands 0.30 Eh lower at −1518.805 Eh.

```python
scf = kuiva.ScalarSCF(metal, stability="follow").run()
scf.stable          # True / False, or None when the analysis was not asked for
```

⚠ `stable is None` is **not** `False`, and it is not `True` either: `if not scf.stable` reads a
run that never measured as unstable, and `if scf.stable` reads it as stable. Compare
explicitly. External stability (RHF → UHF, real → complex) is deliberately not run: the answer
to it is to choose a different `reference=`, which is your decision and not something to do to
you mid-run.

#### Starting from another calculation's orbitals

`guess_from=` takes a finished `ScalarSCF` (or the `ScalarX2CData` it produced) and starts the
SCF from its orbitals. Over the **same AO basis** they are used as they are — the
potential-energy-surface case, where the geometry differs and the basis does not — and over a
**different** basis they are projected onto it, through the same projector `CASSCF(project_from=)`
uses.

```python
small = kuiva.ScalarSCF(mol_svp).run()
big   = kuiva.ScalarSCF(mol_tzvp, guess_from=small).run()      # projected onto the larger basis
```

⚠ **It buys nothing on a closed shell** — measured on Ne, HF and H₂O from `x2c-SVPall-2c` into
`x2c-TZVPall-2c`, the count of Fock builds moves by ±2, which is noise. The saving is real only
where the SCF is hard, and there it comes with a caveat: on TiCl₃ the same projection cut 35
Fock builds to 21 **and landed on a different SCF solution**, 9.7 mEh above the cold start's.

⚠ **So pair it with `stability=` on anything open-shell.** In that TiCl₃ case the solution the
warm start found is internally unstable, and `stability="follow"` walks back to the cold-start
solution to every printed digit. A guess decides *which* stationary point you find; the
stability analysis is what tells you whether it is the one you want.

⚠ It cannot be combined with `init_guess=`, which names a different starting point; and the two
calculations must be the same molecule (elements and their order are checked, the geometry is
not — carrying orbitals along a scan is the ordinary use).

#### Antiferromagnetically coupled centres: the broken-symmetry guess

⚠ **An unrestricted SCF started the ordinary way stays symmetric.** The closed-shell density is
a stationary point of the energy, so nothing in the iteration pushes off it: you ask for UHF on
a coupled pair of metals, get the restricted answer, and nothing in the output says so. The
polarization has to be put in by the starting density.

```python
scf = kuiva.ScalarSCF(dimer, reference="uhf",
                      broken_symmetry={"Fe1": +5, "Fe2": -5}).run()   # signed, per centre
```

The values are **signed counts of unpaired electrons per centre**, addressed the same way a
per-atom basis or reference configuration is (`"Fe1"`, `"Fe"`, or a 1-based atom number). Kuiva
converges the **high-spin** state — the easy, unambiguous one — localizes its singly occupied
orbitals onto the centres you named, flips the ones you asked to be spin-down into the beta set,
and runs the SCF from that density. For two equivalent metals that localization is what makes
the flip expressible at all: the canonical orbitals are the symmetric and antisymmetric
combinations, each half on each metal.

Two things are then reported, and **both** matter:

- **`<S^2>` between the low-spin and high-spin values** says the determinant really is
  broken-symmetry. Coming back at the low-spin value means the polarization did not survive the
  iteration, and Kuiva warns rather than letting it pass as a converged UHF.
- **The spin populations must carry the signs you asked for**
  (`kuiva.props.population.scalar_spin_populations`). A solution with the two centres swapped
  has the same energy and the same `<S^2>`, and is a different state; nothing but the signs can
  tell them apart.

Measured on two Ti(3+) ions 4 Å apart: the ordinary UHF gives `<S^2> = 0` and zero spin on both
metals, and the broken-symmetry guess gives `<S^2> = 1.00`, `+1.00 / −1.00` electrons of spin,
and an energy **0.29 Eh lower**. ⚠ It is one of the three mutually exclusive ways to start an
SCF (with `guess_from=` and `init_guess=`), needs `reference="uhf"`, and refuses when the
magnetic orbitals do not localize — `bs_min_population=` is the knob, and the refusal prints the
populations, because flipping an orbital that is half on the other centre produces a density
that is not the pattern you asked for.

#### The gauge origin, and the one unit trap in this API

It is fixed **here**, not at the property dump, because `L` is defined relative to it and the
multireference layer never calls PySCF again. Five forms:

```python
kuiva.ScalarSCF(mol)                                     # centre of mass, the default
kuiva.ScalarSCF(mol, gauge_origin="charge")              # or "origin"
kuiva.ScalarSCF(mol, gauge_origin=("atom", 1))           # on atom 1 (1-based)
kuiva.ScalarSCF(mol, gauge_origin=("atom", "Dy"))        # ...or name it, if there is one Dy
kuiva.ScalarSCF(mol, gauge_origin=("angstrom", 0, 0, 1.5))     # explicit, in Angstrom
kuiva.ScalarSCF(mol, gauge_origin=("bohr", 0, 0, 2.835))       # explicit, in bohr
```

`("atom", k)` takes a 1-based number, an element symbol, or a label like `"Ti2"` — the same
addressing per-atom bases and reference configurations use, so there is one way to name an
atom in this program. ⚠ An element symbol naming *several* atoms is refused rather than
resolved to the first: "put the origin on the chlorine" is not a statement about a molecule
with three of them.

⚠ **A bare `(x, y, z)` tuple means bohr, and your geometry is in Angstrom.** That is the
historical meaning and it has not changed, so no existing script breaks and no stored number
moves — but a coordinate copied out of the geometry into `gauge_origin=` lands 1.89× too far
out, with no error. It moves the point `L` is defined about, so every orbital moment in the
dump is wrong and every one of them looks entirely reasonable. The bare form therefore
**warns**, naming the two united spellings. Use `("bohr", …)` or `("angstrom", …)` and it
says nothing.

#### Point-group symmetry

`point_group=` on `ScalarSCF` (or on the `Molecule`) turns on **abelian double-group
symmetry**: every orbital gets an irrep label, and the run prints the character table of the
group it is actually using — boson *and* fermion (spinor) irreps, with every operation named
by its lab-frame geometry (`C2(z)`, `sigma(xy)`, `i`) rather than by a Schoenflies label whose
orientation you would have to guess. Two programs agreeing on "C2h, Bu" and disagreeing on
which axis is `z` produce different numbers and no error message; the table is there so that
cannot happen quietly.

```python
scf = kuiva.ScalarSCF(molecule, point_group="auto").run()   # or "C2h", "D2h", ...
```

Labels alone change nothing. What they enable is asked for separately: **per-irrep state
selection** (`n_states={"1E1/2g": 2, "2E1/2g": 2}`) and a **symmetry-preserving orbital
optimization** (`preserve_symmetry=True`), both on `CASSCF`.

Where the molecule's real group is bigger than the abelian one — every atom, and every group
reduced above — the converged states are additionally **classified** by the irreps of the full
point double group, and the run then prints three tables: the abelian one used in the
mathematics, the full one used in the labelling, and the computed correspondence between them.
On planar TiCl3 the abelian labels call all five Kramers doublets of the `d` shell the same
thing; the `D3h` labels separate them into three multiplets. ⚠ This **classifies and never
adapts**: no symmetry-adapted many-particle basis is built, every stage still runs in the
abelian subgroup, and no number changes. What it adds is a refusal — a state count that cuts a
multiplet whose dimension theory fixes. Pass `classification=False` to switch it off, or
`classify=False` on a single `CASCI`/`CASSCF`.

Five things are worth knowing before you switch it on.

- ⚠ **The molecule is never reoriented.** The operations are tested in the frame you give the
  geometry in, because reorienting would move the gauge origin and every property operator
  fixed with it. A symmetry axis that is not `z` is reported and **not used** — orient the
  input so the axis is `z`.
- ⚠ **Groups whose *double* group is not abelian are reduced.** `C2v`, `D2` and `D2h` have
  two-dimensional fermion irreps, which one integer cannot label, so the labels come from the
  largest subgroup that does have one-dimensional ones (`C2(z)`, `C2h(z)`). The reduction is
  reported.
- ⚠ **A per-irrep count is not a safety mechanism.** Where the real group is bigger than the
  abelian one being used — every atom, and every reduced group above — two members of a
  physically degenerate manifold can carry different labels, so a per-irrep count can split
  the manifold exactly as a plain count can. The state-average boundary check stays the thing
  that catches it.
- ⚠ **A single state inside a degenerate block has no irrep**, for the same reason a single
  spinor inside a degenerate manifold has no populations: the eigensolver may return any
  rotation of the block. Blocks are labelled, and a block spanning two irreps is named for
  both.
- ⚠ **A per-irrep count and the Kramers-restricted CI combine over *conjugate pairs* of
  irreps.** Time reversal conjugates a label, so a sector is time-reversal-closed only when it
  is self-conjugate; asking for `n` states of an irrep in that mode returns the `n`
  time-reversed partners in its conjugate as well. That combination also makes well defined a
  request the general path refuses — `n` states of one sector alone splits every Kramers pair,
  because the partners live in the conjugate sector and are not selected.

Degenerate orbitals are **symmetry-adapted** rather than refused: the SCF is free to return
an arbitrary mixture of two degenerate orbitals, and a mixture is an eigenvector of nothing.
The rotation happens inside the degenerate block, so no density, energy or observable
changes, and the run reports how many blocks it touched. An orbital that is still not an
eigenvector afterwards is refused, naming the orbital and the operation.

#### Which centre an active orbital belongs to

Selection by character says *which* orbitals are active; for two equivalent centres it cannot
say *which centre*, and the honest answer for the canonical orbitals is "both". Localization
rotates inside the active space so that every orbital sits on one site:

```python
from kuiva.interface.api import active_space_for, localize_active_space

space = active_space_for(ref, character=([0, 1], "d"), n_active=20, n_active_elec=2)
local = localize_active_space(ref, space, [0, 1])          # one site per metal
cas   = kuiva.CASSCF(ref, active=space, coeff=local.coeff).run()
```

⚠ **It changes no number** — the rotation is active-active, so the CI energy is invariant to
machine precision (measured 5e-15 Eh). What it changes is what the orbitals mean, which is what
a broken-symmetry guess flips, what a multi-centre pseudospin export partitions, and what a
tensor network wants its modes ordered by. Sites are addressed like every other per-atom
feature, a site may be a whole fragment (`[["Fe", 3, 4], ...]`), and `counts=` states an uneven
split. A set that does not localize is **refused** with its populations printed, since a site
partition that is half delocalized is one in name only.

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
scf = kuiva.ScalarSCF(mol, fitting="cholesky-direct").run()          # never store the integrals
```

**Which Cholesky route runs is decided automatically by default** (`fitting="auto"`, the
default): the stored-array route wherever its memory plan fits the configured limit, and
**`fitting="cholesky-direct"`** where it does not — the same decomposition, the same threshold
and the same error bound, with each column of two-electron integrals evaluated when the
pivoting asks for it instead of the whole array being built first. That array grows as the
fourth power of the basis, so it is what decides whether a large system starts at all — which
is why the plan, not a size constant, makes the choice: the two routes cost the same processor
time to within a few per cent on anything beyond ~160 basis functions (the stored route is up
to ~40% faster below that, and always fits there), so the array is the entire decision. When
the automatic choice takes the direct route, the output's "two-electron route" line says so and
why. Either route changes no result — measured identical on a spin-orbit spectrum, and to
3e-15 on the integrals themselves. Pass `fitting="cholesky"` or `fitting="cholesky-direct"` to
pin a route regardless of the plan. ⚠ On the direct route the decomposition happens in the SCF
stage, so `cholesky_tol` and `orbit_pivots` belong to `ScalarSCF` rather than to `Reference`;
passing a different threshold to `Reference` afterwards is reported and not silently applied.

**Where the finished factor rows live is a second, independent choice** (`factors=` on the
SCF stage: `"in-core"`, `"scratch"`, `"streamed"`, or `"auto"`, the default). `"scratch"`
spills the rows to a file in the configured scratch directory right after the decomposition
and streams them back in the same sequential blocks every consumer already reads — freeing
their memory for the CI workspace and orbital-optimization blocks of the later stages, at no
measured cost per pass on a machine whose spare RAM holds the file in the page cache (and at
file-size over disk-bandwidth per pass where it does not). Both produce identical results,
bit for bit.

`"streamed"` goes one step further and runs the **decomposition itself out of core**: each
Cholesky vector is written to the scratch file as it is produced, and the update reads that
file back in sequential passes, so the factor array is never allocated at all. That array is
what otherwise bounds the largest system the front end can start — about 45 GB at 1000 basis
functions — and it is now a working set you can size instead. What it costs is passes over
the file: the number of Cholesky vectors divided by how many candidate columns the working
set holds, measured at 35 passes and 3.5 GB read for a 0.2 GB factorization given a quarter
of its size to work in, for about 10% more processor time than the in-core decomposition.
⚠ Unlike the spill, this is **not** bit-for-bit identical to the in-core route: the same
subtraction is summed in a different order, so the two agree to rounding and, where two
symmetry-equivalent columns are tied, may pick them in the other order — a different but
equally valid factorization of the same matrix, with the same vector count and the same error
bound. Measured on a 3d complex: identical vector count and residual, and Coulomb/exchange
matrices agreeing to 1e-12 absolute, or to the Cholesky threshold when the tie breaks the
other way.

`"auto"` takes each rung exactly when it lowers the planned peak below the memory limit and a
scratch directory is configured, announced on its own output line. ⚠ Density-fitting factors
never spill or stream (they are the ingested container's own array; the request is declined
with a warning).

**The stored route's integral array is released as soon as the factors exist.** Nothing
downstream reads it again, so once `Reference` has factorized it the `O(nao^4/8)` array and
its share of the memory budget are given back — 7.7 GB at 300 basis functions, against factor
rows of 1.2 GB — and the memory plan stops charging every later stage for it. The output says
what was released. ⚠ If you mean to factorize the *same* ingested SCF twice (a threshold
series, say), pass `release_eri=False` to the first call; a second one afterwards is refused
with that advice rather than silently finding the array gone.

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

**Two shells in one active space** — the union form is how you say it, and it is worth spelling
out because the syntax does not announce itself:

```python
character=[("Dy", "f", 14), ("Dy", "d", 10)]     # 4f + 5d, one active space
```

Each triple is `(atom, l, n_spinors)`, in whole Kramers pairs — `14` is the seven f pairs, `10`
the five d pairs. The fragments are resolved independently and unioned, and a pair claimed by
two of them is **refused rather than shared**: a pair clearing two thresholds at once means the
fragments were not the disjoint physical statement they were written as.

⚠ **A second shell of the *same* `l` is a different case**, because two fragments with the
same `(atom, l)` select the same lowest pairs and are refused for overlapping. Either pool
them into one selection, or offset the second fragment with an **ordinal window**:

```python
character=("Ti", "d"), n_active=20                    # the lowest TEN d pairs: 3d + 4d
character=[("Ti", "d", 10), ("Ti", "d", 10, 5)]       # the same twenty, named as two shells
```

⚠ **The same window is what names a valence shell that has filled shells of the same `l`
below it, and forgetting it fails silently.** `character=(atom, l)` takes the **lowest**
qualifying pairs, which is the valence shell only when nothing of that `l` is filled beneath —
true for 3d and 4f, and false for every `p` shell above the second row. `character=("Ga", "p")`
selects Ga's **2p core**, not its 4p valence electron, and the calculation converges and reports
an entirely ordinary spectrum. Two things make this worth stating rather than leaving to the
populations: `g` values cannot detect it, because a `p¹` shell is Landé 2/3 whichever shell it
occupies; and whether the CASSCF repairs the wrong starting guess depends on the Hamiltonian
rather than on the active space — measured, the same Ga selection lands on the valence answer
with `screening="none"` and on a ²P splitting of 249 400 cm⁻¹ (against an experimental 826)
with the default `screening="x2camf"`. **Count the filled shells of that `l` and skip them**
(`("Ga", "p", 6, 6)` — 2p and 3p are six pairs), and check the result: a computed splitting
against a published one is the cheapest check that catches this.

The fourth element of a fragment is `skip_pairs`: how many qualifying pairs to step over
before taking its own. The second form therefore reads *"the five lowest d pairs on Ti, plus
the next five"* — the same twenty spinors as the first, but recorded as two fragments, which
is what a later per-fragment report or a localization step needs. ⚠ Neither form names a
*principal quantum number*, and that refusal is not softened: an `n` label counts shells
within the basis, while an ordinal within a stated character-and-threshold ordering is
something another program can reproduce. Which shells you actually got is still a question
for the populations below — or for AVAS, which answers it by construction.

⚠ **The pre-optimizer will never suggest a double shell.** `CheapCI.suggested_active()` selects
on fractional *occupation*, and the correlating shell of a double-shell active space is empty at
that level of treatment — its members come back at ~1e-4 and are not returned. That is a
structural blindness, not a threshold to lower, and it is the concrete reason the suggestion is
documented as a lower bound: a double shell has to be asked for.

#### `avas=` — when no single orbital *is* the target shell

A character selection can only pick orbitals that already carry the character. Where the
metal–ligand bond is covalent the d (or f) weight is spread over several bonding and
antibonding combinations, none of which clears any threshold, and the active space you want is
a **rotation** of them. That is what AVAS constructs:

```python
scf = kuiva.ScalarSCF(mol, atomic_reference=True).run()      # AVAS projects onto these
cas = kuiva.CASSCF(kuiva.Reference(scf).run(),
                   avas=dict(atom="Ti", l="d")).run()
cas.avas.report()          # the projection eigenvalues, and the gap at the cut
```

Every orbital is projected onto the free-atom valence orbitals of `(atom, l)`, the projector is
diagonalized within the occupied and within the virtual space, and the combinations of large
eigenvalue become the active space. `active=`, `character=` and `avas=` are three ways of
answering one question and exactly one may be given.

* `threshold=` (default 0.2) is the projection eigenvalue a Kramers pair must carry. ⚠ It is a
  **selection knob, not a tolerance**: the right value is the one that falls in the gap of the
  eigenvalue spectrum, which the report prints for exactly that reason. A small gap is warned
  about, because it means the threshold and not the electronic structure chose the space.
* `n_shells=2` is the **double shell** — the target shell plus its correlating partner. This is
  the case a character threshold cannot find at all, the correlating shell being diffuse and
  covalent.
* `max_pairs=` refuses rather than returning more than you expected. Worth setting: a threshold
  slightly too low gives a perfectly plausible space one or two pairs too large, and the cost of
  that is discovered when the CI runs.

⚠ **Three things to know.** Kuiva projects onto the free-atom orbitals `atomic_reference=True`
already computed, at the same per-element reference state the atomic mean field uses (neutral;
M(3+) on the f block) — **not** onto the published method's minimal MINAO basis. The orbitals
selected agree; the eigenvalues are not numerically comparable with another program's AVAS.
The rotation stays inside groups of *equal occupation*, so the SCF density and energy do not
move at all. And an AVAS space carries **no symmetry labels**: they belong to the guess spinors
and AVAS has rotated them, so per-irrep `n_states` is unavailable after it — select by
character if the labels are what the run needs.

Put `avas=` on a `CheapCI` if a pre-optimization is wanted; it is refused on a `CheapCI`
*upstream*, because the cheap CI's natural occupations are all distinct and the rotation would
be the identity.

The states: `n_states`, `weights`. The optimizer: `mode` (`"auto"`, `"quasi-newton"`,
`"second-order"`), `max_iter`, `conv_grad`, `conv_energy`, `max_step`, `callback`.
Checkpointing: `checkpoint=path`, `restart=path`, `checkpoint_options`. Stopping in time:
`deadline=`, `signals=` (below).

#### After the run: `<S^2>`, and a term label with its evidence

```python
cas.spin_analysis().report()      # <S^2> per degenerate block
cas.assign()                      # the label, the evidence, the fit residual
```

`spin_analysis()` works on **both** solver routes through one implementation: the CI solver
applies `S` to its vectors — one contraction of the excitation map it already builds — and the
tensor-network solver contracts the same quantities through each root's own densities. With spin–orbit coupling **off** the block value is
`S(S+1)` exactly and `2S+1` is the term multiplicity, read straight off. With it **on** `S` is
not conserved, and the same number measures how pure the spin of a level still is: how much a
`^6H_{15/2}` label is really worth. ⚠ It is reported **per degenerate block and never per
state** — inside a block the eigensolver may return any mixture of the members, so one state's
value belongs to that arbitrary choice while the block trace does not. The report also states
how much of the value came from `S` reaching *out* of the active space, which is computed and
included rather than assumed away.

⚠ **Do not expect an exactly integral `2S+1` from a spin–orbit run, even for a one-electron
active space.** A converged general-complex CASSCF has orbitals that are not spin-pure —
mixing spin is what spin–orbit coupling does — so the state carries a little contamination and
`S` reaches out of the active space. TiCl₃'s `3d¹` doublets come out at `⟨S²⟩ = 0.7511` rather
than 0.7500, with the out-of-space contribution measured at 1.7e-03 beside them; a free boron
atom, whose orbitals are very nearly spin-pure, gives 0.7500 to six figures. The excess is a
measurement of the mixing, which is exactly what the number is for on this path — read it
against the leakage line printed with it, not against an integer.

`assign()` offers a `^{2S+1}L_J` label per block, inferred from the block dimension (`2J+1`),
`<S^2>` (`S`) and the measured isotropic g with the Landé formula inverted for `L`.
⚠ **Every label it prints is an inference**, which is why it is its own report with the
evidence and a fit residual beside each row, and never a column of the state table. A block
whose evidence does not add up is labelled `?` rather than given the nearest plausible term —
which is the normal and correct outcome for the crystal-field levels of a complex, since those
are not `2J+1` manifolds at all. On the tensor-network route the moment matrices behind the g
evidence come from transition densities contracted through the network (an applied-string Gram,
phase-arbitrary exactly as the CI's are), so both calls work on either solver.

#### ⚠ Before you set `n_states`: four questions

The reasoning is in [A first calculation](#a-first-calculation) item 3. This is the short form,
put where the number is actually set — the state average is the single most common way to get a
plausible wrong answer out of this program.

1. **Does the count land on a manifold boundary?** A count that stops *inside* a near-degenerate
   manifold makes the averaged density non-invariant; the Fock operator built from it splits the
   shell, and the selection keeps cutting the same way. Kuiva solves a few roots the average does
   *not* use and reports the gap at the starting orbitals and at the converged ones, warning
   below 50 cm⁻¹. Read those two lines — they are the only evidence there is.
2. **Is it the count for *this* spectrum?** A count that bounds the `2S+1` terms of a spin-free
   calculation generally cuts the `2J+1` multiplets of the same system with spin–orbit coupling
   on, and the converse bites equally. Carry the two counts separately; do not reuse one.
3. **Is the average one the symmetry leaves invariant?** Landing on a boundary is necessary and
   not sufficient. The converged boundary report also states the averaged density's **spin
   non-invariance**: near zero — a whole term, or the whole space — is safe by symmetry and the
   gap stops mattering; large — a single J, a single Kramers doublet — means the average *leans*
   on the spin–orbit structure and is protected only by its gap and by where it started.
4. **If it leans, where did the orbitals come from?** Converge the whole-term (or full-space)
   average first, start the sub-manifold run from those orbitals, and check the manifold
   splitting of the answer directly. A leaning average can converge from the scalar guess into a
   wrong basin with every diagnostic clean, and no static check can see that from inside the run.

`boundary_check=0` switches the diagnostic off; `boundary_check=n` sets how many extra roots it
solves. ⚠ It never kills a run: a failure to *measure* the gap is a warning and **no** report,
which is a weaker statement than a clean one, not a substitute for it.

A count that would split a Kramers pair is refused outright, as is one that cuts a multiplet
whose dimension the full double group fixes (with `point_group=` on). Neither refusal catches a
manifold cut in half — an even remainder passes both — which is why the questions above are
yours and not the program's.

With point-group symmetry on, `n_states` may be a mapping instead of a count, and the
orbital rotation may be constrained:

```python
cas = kuiva.CASSCF(ref, character=("Ti", "d"), n_active=10,
                   n_states={"1E1/2": 2, "2E1/2": 2},   # per irrep, not "lowest n"
                   preserve_symmetry=True).run()        # rotations within an irrep only
```

Each irrep is solved in its own sector of the determinant space, so *"the two lowest states
of `1E1/2`"* is a request rather than a filter applied afterwards — and it is a request the
plain "lowest n" form cannot express at all. The states come back merged and ascending in
the ordinary convention, so the state-averaging gate, the boundary diagnostic, the property
dump and NEVPT2 see exactly what they always did. ⚠ A per-irrep spectrum is only meaningful
while the orbitals are symmetry-pure, which is checked at every solve rather than assumed.

`preserve_symmetry=True` masks inter-irrep rotations out of the parameter list, so the
labels are exact at convergence rather than only at the start. ⚠ It is a **constraint**: what
it converges to is the lowest *symmetric* solution, which is not the global one wherever the
symmetry is spontaneously broken. Without it the drift is measured and reported rather than
assumed away.

The tensor-network solver takes the same labels as a **conserved quantum number** rather than
a classification (`solver_options=dict(symmetry=..., sector="1E1/2")`), so a labelled sweep
cannot leave the sector it targets.

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
  upstream. `checkpoint=`/`restart=` work here too and write **two** files: the ordinary
  trajectory checkpoint (orbitals, RDMs, optimizer state; no CI vectors — the network has
  none), and a sibling `*.network.h5` holding the network state itself, written rolling at
  the end of each completed sweep. A restart resumes the trajectory exactly and warm-starts
  the network from the sibling; an absent or unfitting sibling warns and the first solve
  starts cold, which costs time and never correctness. `restart=` needs `adaptive=False`.

  **The environment cache pages to scratch under memory pressure** (default on). The cache —
  one `D^2 x D_op` tensor per directed bond — is the solver's dominant memory object at
  large bond dimension, while a two-site update only ever touches a handful; when an
  environment's reservation would be refused, the coldest entries move to a scratch file
  and stream back on demand, bit for bit. A solve whose environments fit never touches
  scratch, and a machine with no scratch directory configured gets the refusal it always
  got, naming the knob. `environment_resident_gb` (on `solve_ttn`) additionally caps the
  resident set below the memory limit.

  **Production controls**, all in `solver_options`: `bond_schedule=[16, 32, 64]` ramps the
  cap per sweep inside the first solve (the manifold is the final cap; convergence is only
  declared at it); `expansion=1e-4` perturbs its truncations to escape a local minimum — the
  *deterministic* subspace expansion of Hubig et al. (2015), which is White's density-matrix
  noise (2005) evaluated instead of sampled, chosen because the deterministic term keeps
  degenerate Schmidt groups exactly degenerate so the group-complete truncation still means
  what it says, and because no RNG enters any trajectory (it decays off over the first
  sweeps; energies stay variational at every strength; the reported `w_disc` is always the
  ensemble's own); and `bond_steps=[64, 128, 256]` is the per-macro-iteration cap ladder —
  each rung is a chart change offered through the propose/adopt seam and adopted only when
  it lowers the energy at fixed integrals, so giving it selects the event-gated driver
  automatically, and a rung refused for gaining too little is a measurement that the
  current cap suffices. The `E(w_disc -> 0)` extrapolation — the number a tensor-network
  paper quotes — is a separate driver over a converged problem: `kuiva.dmrg.bond_series`
  runs a warm-started ascending-D series at fixed integrals and fits each root, and it
  reports the extrapolate **with the series and its fit residual beside it, never alone**
  (an exact series is reported exact, and a residual comparable to the claimed correction
  warns that the series is not in the linear regime).

  ⚠ **Read `w_disc`.** The iteration table gains a column carrying the largest discarded
  weight of that macro-iteration's sweeps, and the stage report gives the final value; both
  are also on the finished stage as `cas.max_discarded`. It is the network's primary quality
  number — the fraction of the state the truncation threw away — and every energy from this
  solver has to be quoted with it. The *trend* matters as much as the value: truncation
  growing as the orbitals move is the signal that `max_bond` is too small. The per-sweep table
  stays at DEBUG, because one table per sweep across many macro-iterations would bury the
  output file it is printed into.

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

⚠ **The state average is refused on the same grounds.** `n_states` and `weights` are recorded
in the checkpoint and cross-checked, and a restart asking for a different average is refused
rather than continued: a different state average is a different *calculation*, not a different
chart of this one — the energy functional itself changes, so the converged orbitals and energy
do. Continuing from converged orbitals into a **new** state average is a real thing to want,
and it is `coeff=`, not `restart=`. (A checkpoint written before the average was recorded
cannot be checked, and says so rather than passing quietly.)

#### Starting from a calculation in a different basis set

A CASSCF costs what its basis costs, and almost all of that expense buys the *orbitals* —
while the active space, which is a statement about chemistry, is very nearly the same in a
small basis as in a large one. `project_from=` converges the calculation where it is cheap
and continues it where you want it:

```python
cheap = kuiva.CASSCF(kuiva.Reference(scf_small).run(),
                     character=("Ti", "d"), n_active=10, n_active_elec=1).run()

cas   = kuiva.CASSCF(kuiva.Reference(scf_large).run(), project_from=cheap).run()
```

The converged spinors are re-expressed over the target AO basis, made orthonormal again, and
the dimensions the larger basis adds are completed from its own SCF. The source may be a
`CASSCF`, a `CheapCI` or a plain `Reference`, and it may be in a *larger* basis than the
target — projecting downwards is the same call.

⚠ **The active space comes across with the orbitals and may not be restated.** It was chosen
once, against the orbitals being carried; re-selecting it here would resolve it against the
target reference's own guess orbitals, which is a different calculation wearing the same
name. (A projection from a bare `Reference` has no active space to inherit, so there it *is*
stated — and is then resolved against the source.) A projection replaces the `CheapCI`
pre-optimization rather than following it, and does not combine with `restart=`.

`projection=dict(...)` configures it:

- **`carry`** — `"active"` (default) carries the active orbitals and takes the inactive and
  virtual ones from the target's own SCF; `"all"` carries every orbital. ⚠ The default
  deliberately carries *less*. The source's inactive orbitals are not eigenvectors of
  anything in the target basis, so carrying them re-introduces an inactive–virtual gradient —
  where the core orbital energies are the largest numbers — that the target's own SCF had
  already removed. Measured, `"all"` costs roughly twice the macro-iterations.
- **`scheme`** — how the projected set is made orthonormal again: `"blocked"` (default,
  space by space, symmetric within each, so the CAS partition stays exact), `"symmetric"`
  (one Löwdin over the whole set: the smallest total change, and the only one allowed to mix
  active with inactive and virtual character) or `"gram-schmidt"` (never inverts a Gram
  matrix, so it is the most forgiving of a heavily truncating downward projection).
- **`repair_pairing`** — rebuild the carried orbitals as explicit Kramers pairs, on by
  default. ⚠ A converged general-complex CASSCF is entitled to leave its *active* orbitals
  far from pair-aligned, because rotations inside the active space are redundant and nothing
  pushes back; everything downstream that reads the pairing convention needs pairs.

The projection reports what it did, and the numbers are invariants rather than coefficient
comparisons: the **retained norm** of each source orbital (how much of it exists in the target
basis at all), the **principal overlaps** between the source active space and the one handed
on (unchanged by any rotation inside either space), and the **complement separation** (the
evidence that the orbitals with no source really are the orthogonal complement). ⚠ Read them.
Every way of getting a basis projection wrong still produces an orthonormal orbital set of
the right shape that starts a calculation which converges — the failure shows up only as a
run that takes longer than it should.

Example 9 is this workflow end to end. The lower-level entry point is
`kuiva.interface.api.project_to_basis`, which returns the projected coefficients and the
diagnostics without running anything.

### `CASCI(upstream, ...)` — a spectrum at fixed orbitals

A full CI over the active space at orbitals that are **not** re-optimized. `upstream` is any
finished stage that carries orbitals — a `CASSCF` (the usual one), a `CheapCI`, or a plain
`Reference` for the CI at the SCF guess — and the orbitals and the active space come from it.
This is where a scan is written: what varies is the number of states, a per-irrep request, the
CI symmetry mode, a Davidson tolerance, or the active space restated against those same
orbitals.

```python
cas      = kuiva.CASSCF(ref, character=("Ti", "d"), n_active=10,
                        n_active_elec=1, n_states=2).run()   # orbitals for the ground doublet
spectrum = kuiva.CASCI(cas, n_states=10).run()               # all ten, at those orbitals
spectrum = kuiva.CASCI(cas, n_states=10,
                       solver_options=dict(kramers="restricted")).run()   # same ten, cheaper
```

Optimizing the orbitals for the states you want and then reading the full spectrum off them is
the ordinary shape of a ligand-field calculation, and it is what the second line above costs:
one CI solve instead of a second orbital optimization.

⚠ **Two rules, and they are one rule stated twice: a statement about orbitals belongs to the
orbitals it was made against.**

- `character=` and `avas=` read atomic populations off the **reference's own SCF orbitals**, so
  they may be stated only on a `Reference` upstream, where those are also the orbitals the CI
  runs at. On a `CheapCI` or `CASSCF` upstream the space is inherited, and an active space
  varied at fixed orbitals is stated as `active=[spinor indices]` — a statement about the
  orbital set at hand. The orbitals have moved, so re-running a character selection against
  them may legitimately return a *different* set, and the spectrum would then not be the one
  belonging to those orbitals, with nothing in the output saying so. Restating it is refused
  rather than reconciled.
- `coeff=` is accepted only where there is nothing to inherit — on a `Reference` upstream, for
  orbitals that came from somewhere else — and then together with `active=` for the same
  reason. Elsewhere the chain already answers *"which orbitals"*, and two answers to that is
  how a state set and an orbital set stop matching.

The state-averaging gate applies exactly as it does to a `CASSCF`: weights are equalized inside
a degenerate block and a count that splits one is refused. What does not run is the
state-average **boundary diagnostic** — that is a statement about an orbital trajectory, and
there is none here. ⚠ A CASCI energy is variational at those orbitals and nothing more: over a
state average the upstream CASSCF did not optimize, the orbitals are not stationary for it, and
the levels can order themselves differently from a CASSCF over the same states.

It feeds `NEVPT2` and `PropertyDump` exactly as a `CASSCF` does — but not `PseudospinExport`,
which consumes converged *orbitals* rather than states and so belongs on the stage that
optimized them.

The function underneath is `kuiva.interface.api.casci`, which takes the orbitals as `coeff=`
and is what to drive when there is no stage to hang them on.

### `NEVPT2(source, **options)`

Strongly contracted NEVPT2 on a converged reference — post-processing, per state, decomposed by
excitation class. It consumes the converged orbitals and CI vectors and changes no
wavefunction. `source` is a finished `CASSCF` or `CASCI`; ⚠ on a `CASCI` the total is
`E(CASCI) + E2`, a different reference from `E(CASSCF) + E2` and not comparable with it.
Options: `frozen_core`, `deleted_virtual`, `shift`, `imaginary_shift`, `fock`,
`classes`.

```python
pt = kuiva.NEVPT2(cas, frozen_core=-10.0).run()   # an orbital ENERGY, never a count
pt.multiplets()                                   # barycentres beside the per-state energies
```

- **Frozen core and deleted virtuals are off by default** and are stated as an orbital energy
  on the pseudo-canonical spectrum, never as a count.
- Intruder-state level shifts (real and imaginary) exist and are **parameter-free by default**;
  any applied shift warns.
- ⚠ **The intruder diagnostic warns rather than merely printing.** Each class's smallest energy
  denominator appears in its table as `min |dE|`, and it is now compared against a fixed band:
  below **0.1 Eh** that class's `E2` leans on a perturber contributing out of proportion to its
  physical weight, and Kuiva says so, naming the class and pointing at `imaginary_shift`. Below
  **1e-6 Eh**, or at a **negative** denominator — a perturber that has fallen to or below the
  reference — the warning is louder, because the class energy is then divergent rather than
  ill-conditioned, and a negative denominator makes it wrong in sign as well. A level shift
  already applied does not silence either: it bounds the damage, it does not remove the
  intruder. The fix is the reference — the active space, or which states it averages over.
- All eight excitation classes are a *partition* of the first-order interacting space, so a
  restricted `classes=` gives a **partial** `E2` and says so.
- ⚠ **Inside a degenerate CI manifold the individual per-state `E2` depend on the
  eigensolver's arbitrary basis; the barycentre does not.** No contraction fixes this, so the
  treatment is reporting rather than repair: `multiplets()` gives barycentres *beside* the
  per-state energies, with the member spread visible.
- Works on both solver routes: the conventional-CI CASSCF supplies its stored CI vectors;
  a `solver="dmrg"` CASSCF goes through the network-backed contraction provider — ⚠ six of
  the eight classes today, so its `E2` is a loud **partial** sum (see *Limitations*).

### `PropertyDump(source, path, ...)` and `PseudospinExport(casscf, path, ...)`

The two formatted products; see [What a run writes](#what-a-run-writes) for the files
themselves.

```python
kuiva.PropertyDump(cas, "ticl3.props", title="TiCl3 d1").run()
kuiva.PropertyDump(pt, "ticl3.props").run()      # NEVPT2-corrected H; hybrid protocol recorded
kuiva.PropertyDump(cas, "ticl3.props", include_dipole=False).run()   # H and mu only

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

### Driving one piece by hand

The stage classes are a thin layer over `kuiva.interface.api` (`api` below) and the module
drivers, all of which stay public and usable directly. Collected here are the entry points that
have **no stage of their own**, because otherwise they are findable only by knowing the module
path. Reaching for one is a deliberate step off the documented path; each is still bound by the
same memory limit, the same provenance records and the same state-averaging gate as a stage,
being the same code underneath.

(The `kuiva.` top level itself stays thin on purpose. Besides `Molecule` and the eight stage
classes it carries only the *read* counterparts of what those stages write — `read_dump`,
`read_pseudospin`, `read_checkpoint`, `PropertyMatrices`, `PseudospinModel` — because reading
a stored product back is how two calculations get compared at all.)

| call | what it is for |
|---|---|
| `api.property_matrices(reference, source)` | the `H` and `mu` matrices **without writing a file**, for comparing two calculations in memory through `.analyse()`. `api.property_dump` is this plus the write. |
| `api.active_space_for(reference, character=..., n_active=...)` | resolve an active-space request into the object the drivers take, so it can be inspected or reused before a run is committed to. |
| `api.avas_active_space(reference, atom=..., l=...)` | the AVAS space **and its rotated orbitals**, so the projection eigenvalues can be looked at before a run is committed to a threshold. The stage form is `avas=`. |
| `api.spin_analysis(reference, source)`, `api.assign_states(reference, source)` | `<S^2>` per degenerate block, and the term-assignment offer built on it, for a result produced by hand — a bare `api.casci` or a driver call, which has no stage to call `.spin_analysis()` / `.assign()` on. |
| `api.projected_active_space(plan, target, n_active_elec)` | the target-basis active space a projection lands on, for driving `api.project_to_basis` by hand instead of through `project_from=`. |
| `SpinorReference.h_one_electron()` | the `(2·nao, 2·nao)` one-electron Hamiltonian in the AO basis: the full two-component X2C operator with spin–orbit coupling ingested, and the spin-free one lifted to two components without it. This is the operator the correlated energy is an expectation value of. |
| `api.memory_plan(nao, ...)` | the phase-by-phase memory estimate the pre-flight prints, as data. Every entry is a function of dimensions only, so it answers "will this fit?" before any array — or any SCF — exists. |
| `api.build_mole(molecule)` | the PySCF `Mole` a `Molecule` resolves to, including the decorated atom labels (`"Ti2"`) that per-atom bases and reference configurations are addressed by. |
| `kuiva.interface.pyscf_bridge.run_scalar_aoc(element, configuration, basis=...)` | a scalar X2C SCF on **one atom or ion, averaged over a configuration** — spherical, one radial function per shell. The reference an atomic *shell* quantity has to come from, and what `kuiva.extras` is built on. ⚠ Its occupations are fractional, and the pipeline stages are not validated on such a reference. |
| `Molecule.from_xyz_file(path, basis)` | a molecule from an XMol `.xyz` file — count line, comment line, then the atoms, in Angstrom. ⚠ The count is **checked, not trusted**: a file whose header disagrees with its contents is truncated or concatenated, and reading its first *n* atoms would be a different molecule. A headerless file is accepted. |
| `Molecule.from_xyz_string(xyz, basis)` | the same from `"El x y z"` lines held in a string — those lines **only**, no header. Handed a real file it says so and names `from_xyz_file`. |

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

#### Gaunt and Breit: a third axis with no name on it

The four-component reference behind `"x2camf"` and `"mmf"` runs with the Dirac–**Coulomb**
operator by default. The Gaunt and full Breit interactions are available, and
`screening_options` is the only route to them:

```python
scf = kuiva.ScalarSCF(mol, screening_options={"interaction": "gaunt"}).run()   # or "breit"
```

⚠ **The interaction is not part of the Hamiltonian's name.** The method table above has five
names and none of them says which two-electron interaction the reference was solved with: a
Gaunt-corrected run is still called `X2C-AMF`, and two runs differing only in this print the
same method line. What records it is the **screening record's `interaction` field**, which
`scf.data.soc.provenance()` returns and every stored product carries in its header. When two
calculations disagree, compare that field rather than the method name.

`screening_options` also carries `backend` and `uncontract`; the full list is in
`kuiva.amf.correction.amf_correction`.

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

### The environment: point charges

A single-molecule magnet is measured in a crystal, and a bare gas-phase 3+ ion has a
qualitatively different ligand field from the one in the lattice. Point charges are the
smallest honest way to say what surrounds the molecule:

```python
env = kuiva.Environment(
    point_charges=[(+2.0, (0.0, 0.0, 5.4)), (-1.0, (0.0, 3.1, -2.2))],  # (q, (x, y, z))
    label="two shells of the lattice, Ewald-fitted elsewhere")
mol = kuiva.Molecule(atoms=[...], basis="x2c-TZVPall-2c", environment=env)
```

A whole lattice is given as arrays rather than as tuples:
`Environment(point_charges=(charges, coords))`.

- **Coordinates are in the molecule's own `unit`** unless the environment states its own
  (`Environment(..., unit="bohr")`). ⚠ A charge field copied from a crystallographic file is
  in Angstrom; the same numbers read as bohr put the lattice 1.9x too far away and produce a
  perfectly plausible ligand field for the wrong crystal.
- **What it changes is exactly two things**: the one-electron Hamiltonian gains the charges'
  potential, and the classical charge–nucleus energy is added to the total. Everything after
  the front end is untouched — the multireference layer is handed a one-electron Hamiltonian,
  as it always was, so every invariant and every stored product means what it meant before.
- **The charge–nucleus energy is reported on its own line** and is *not* folded into the
  nuclear repulsion, so an embedded total stays separable into the part that is chemistry and
  the part that is the field. ⚠ The charges' interaction with *each other* is a constant of the
  lattice, not of this calculation, and is neither computed nor reported.
- ⚠ **The gauge origin does not move**: charges have no mass and no nucleus, so a complex in
  vacuum and the same complex embedded share one origin and their property files stay
  comparable.
- ⚠ **Symmetry sees the field.** A field of lower symmetry than the nuclei restricts the labels
  (and says which operations it removed), and it switches the non-abelian classification off —
  that layer verifies the *molecule's* full point group, which a field can break in ways the
  three tested operations do not cover.
- **The potential is added bare** — the non-relativistic operator, used unchanged in the
  two-component basis. `Environment(..., picture_change=True)` transforms it through the
  Hamiltonian's own X2C decoupling instead; measured at ~1e-5 relative and **not** growing
  with Z, at the cost of an integral per charge (a lattice makes that the expensive route).
- **The field is recorded in the provenance** by count, net charge, extent and a **digest** —
  a lattice does not belong in a file header, but two files carrying the same digest were
  embedded in the same field.
- ⚠ A charge closer than 0.5 bohr to a nucleus is **refused**: the density polarizes onto it
  without bound and the SCF converges to a number that means nothing.

- ⚠ **What an embedding is worth, measured, and it differs by shell.** For a 3d complex
  (TiCl3 in a model +-1 lattice) the field moves ligand-field levels by up to 926 cm^-1 and the
  ground doublet's g by <=0.17, lifting its axial `gx = gy` degeneracy. For a 4f complex
  (CeCl3) it moves the levels ten times *less* — 89 cm^-1, the shielded shell — but **rewrites
  the ground doublet's g tensor**: (0.880, 2.531, 2.531) -> (0.709, 1.361, 3.501). A
  lanthanide doublet's composition inside its `2J+1` manifold is what the crystal field fixes,
  so **a gas-phase Ln g tensor is a different quantity, not an approximate one**.
- ⚠ **How big the field must be is a property of the CUT, not of the radius.** Truncating an
  ionic lattice on a **sphere never converges** — measured, +-100 cm^-1 out to 2900 charges,
  with no trend, because the surface multipoles do not vanish with radius. An **Evjen-weighted
  cube** (face 1/2, edge 1/4, corner 1/8) is converged to a few cm^-1 at **~300 charges** and
  to 0.1 cm^-1 by 2200. Check the convergence of the lattice's own potential and field gradient
  at the metal before trusting a spectrum: it is free and it shows the same split.
- **Cost is not a consideration at any realistic size**: the bare potential is one batched grid
  integral, measured at **41 ms per 1000 charges**, so a 12 000-charge lattice adds 0.5 s to a
  3 s ingestion.

Generating a charge field — an Ewald fit from a crystal structure — is **not** part of Kuiva;
the list is the input.

### Ghost atoms

An atom written `("ghost-Cl", pos)` carries chlorine's basis functions and nothing else: no
nucleus, no electrons, no mass. It is what a counterpoise correction is made of.

```python
alone   = kuiva.Molecule(atoms=[("Ne", (0, 0, 0))], basis=B)
ghosted = kuiva.Molecule(atoms=[("Ne", (0, 0, 0)), ("ghost-Ar", (0, 0, 4))],
                         basis={"Ne": B, "ghost-Ar": B})       # same electrons, more functions
```

- **The label is the address.** `basis={"ghost-Ar": ...}` reaches it and `basis={"Ar": ...}`
  does not — a ghost and a real atom of one element are different things to everything
  downstream. Two ghosts of one element get decorated labels (`ghost-Ar1`, `ghost-Ar2`) and can
  carry different bases, exactly as two real atoms can. `X-Cl` and `ghost:Cl` are accepted and
  normalized to `ghost-Cl`.
- **It has no chemistry**: no atomic mean field (the two-electron spin-orbit correction skips
  it, and its block is exactly zero), no free-atom reference, no oxidation state — stating a
  reference configuration for one is refused rather than ignored.
- **It costs what its functions cost.** A ghost enters the working basis, the integrals, the
  memory plan and the symmetry detection like any other centre.
- **The atomic-reference charge report gives it a "charge"** equal to minus the density that
  leaked onto its functions — the superposition error resolved per centre.

### A basis the registry does not have

```python
custom = kuiva.CustomBasis(open("my-set.nwchem").read(),      # or {"Ce": parsed_shells}
                           relativistic_treatment="x2c-2c",
                           name="modified-SVP", notes="from Table 2 of the paper")
mol = kuiva.Molecule(atoms=[...], basis={"Ce": custom, "Cl": "x2c-SVPall-2c"})
```

Both an NWChem-format string (what Basis Set Exchange emits) and an already-parsed
specification are accepted, per atom, mixable with registered families.

- ⚠ **`relativistic_treatment=` is required**, because it is the one property that cannot be
  measured from a list of exponents and the one whose absence stays invisible until it reaches
  a heavy element. It takes part in the same cross-atom compatibility check registered
  families go through, so a non-relativistic set alongside an X2C one is still refused.
- **Contraction type is measured from the data**, never declared, and appears in the output as
  what it actually is.
- **Conditioning is unknown and says so**, which routes the two-electron integrals to Cholesky
  — the error bound the user sets rather than an unbounded fitting error.
- Everything else applies unchanged: the set is decontracted for the four-component atomic
  solve, its mean field is cached on the **content** of the shells rather than on the name, and
  a Cartesian basis is still refused.

### The nuclear charge model

The nucleus is a **point charge by default**, which is what every reference number shipped with
this program was produced with. The alternative is the finite Gaussian distribution of Visscher
and Dyall:

```python
mol = kuiva.Molecule(atoms=[...], basis="x2c-TZVPall-2c", nuclear_model="gaussian")
```

- **One statement for the whole molecule, and every consumer inherits it** — the molecular
  integrals, the four-component atomic solves behind `screening="x2camf"`, the free-atom
  reference orbitals behind the atomic-reference charges, and the isolated-fragment blocks of
  the DLU decoupling. There is deliberately no per-atom form: a molecule with one point nucleus
  among finite ones is not a physical model, and the atomic mean field would then be corrected
  against a nucleus the molecule does not have.
- **What it changes.** The effect is concentrated at the nucleus and grows steeply with `Z`:
  it shrinks j-splittings by under 1e-6 relative at neon, ~5e-4 at krypton and ~3e-3 at
  mercury, and it is beyond the resolution of any basis for hydrogen. ⚠ Those are core-region
  operator measurements; what a finite nucleus does to a *valence* property — a g factor, a
  ligand-field splitting — has not been measured here, and the numbers above bound none of it.
- **The model is recorded** in the Hamiltonian provenance, so it appears in the property dump's
  header, in the checkpoint metadata and in the output's Hamiltonian block. A file that does
  not say which nucleus it describes is not comparable with anything.
- ⚠ **It is part of the Hamiltonian, so results are not comparable across settings**, and the
  atomic mean-field cache keys on it: switching the model does not reuse cached corrections, and
  a lanthanide will pay for its four-component solve again.
- ⚠ `screening="x2camf-external"` **refuses** a finite nucleus. The external plugin solves its
  own atomic references over a point nucleus, and it exists to bisect a disagreement — a
  bisection tool that silently answers a different question is worse than one that is
  unavailable.

For an atomic average-of-configuration run there is no `Molecule`, so the model is an argument:
`kuiva.interface.pyscf_bridge.run_scalar_aoc(..., nuclear_model="gaussian")`.

### ⚠ If you are comparing against another program

Everything below is a real difference that will show up as an apparent "method error" if it is
not matched. They are listed because they are invisible in the output otherwise.

- **Gas phase unless you say otherwise.** Kuiva computes an isolated molecule; a reference
  number measured in a crystal, or computed by a program with an embedding switched on, is a
  different system. `Environment(point_charges=...)` is how to say so, and the header of every
  stored product says whether it was used.
- **Point nucleus by default**, and it is now selectable: `nuclear_model="gaussian"` on the
  `Molecule` puts the Visscher–Dyall finite nucleus under every integral instead (see
  [The nuclear charge model](#the-nuclear-charge-model)). DIRAC defaults to the Gaussian one,
  so this is usually the first thing to match, and the difference is a genuine physical effect
  that **grows with Z**: below 1e-6 relative on a neon j-splitting, ~3e-3 on mercury's.
  ⚠ **Those are CORE numbers and a valence property moves ~40x less** (measured): the
  *valence* 2P splitting of Tl (Z = 81) shifts by 6.8e-05 relative, and a ground-doublet **g
  factor never moves by more than 1.2e-05** anywhere from B to CeCl3 — two to six orders below
  the error the method already has against an analytic Lande g. So the option is worth turning
  on for one identified case, a heavy-element **ligand-field** spectrum quoted to better than
  ~0.1 cm^-1 absolute or ~2e-04 relative (CeCl3's crystal-field splittings shift by 0.111 cm^-1,
  and a ligand field amplifies the free ion's effect ~35x) — and it is never what fixes a g
  factor. ⚠ It costs a second four-component atomic solve per element (~40 min for Tl or Ce on
  a laptop-class box), paid once ever and cached.
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
changes no scalar quantity. ⚠ It does **not** turn spin–orbit coupling off; that is
`with_soc=False`, and the two are described at the end of this section.

The reference configuration defaults to the neutral atom, except the f block, which defaults to
M(3+), on chemistry. Open shells are occupied by average of configuration, not aufbau.

**Choosing reference states is a first-class argument, one statement per atom.**
`ScalarSCF(mol, configuration=...)` takes a mapping whose keys are element symbols (`"Ti"`),
atom labels (`"Ti2"`), or 1-based atom numbers (`3`) — most specific wins — and it feeds
*both* consumers of a reference state at once: the atomic mean field and the atomic-reference
charges. Each value is either an **oxidation state** (`"+3"`, `2`, `-1`) or an **explicit
configuration** (`"[Xe]4f1"`, `"1s2 2s2 2p5"`). An oxidation state resolves through a curated
table of each element's most common states to exactly **one canonical configuration** — a
state outside the table warns and falls back to the standard derivation rule — while an
explicit configuration is checked against the table's *accepted* set: where the literature
genuinely admits more than one occupation (the d/s ambiguities of the late transition metals,
say Ni's 3d⁸4s² against 3d⁹4s¹), every accepted one passes silently, and anything else warns
as an excited or unusual reference. Impossible configurations (a channel no shell of the
element's period can hold, an anion closing no shell when derived from an oxidation state) are
refused outright. Two atoms of the same element may carry **different** states — they then get
decorated labels (`"O1"`, `"O2"`) throughout output and provenance — and a scalar
`configuration` on a heteronuclear molecule still raises, because "+3" almost always means
"the metal is trivalent". The older `screening_options={"configuration": ...}` spelling keeps
working and warns when both are given.

**Different bases for different atoms use the same addressing.** `Molecule(...,
basis={"default": "x2c-SVPall-2c", "O": "x2c-TZVPall-2c"})` upgrades every oxygen;
`basis={"default": ..., 3: "x2c-TZVPall-2c", 15: ...}` upgrades atoms 3 and 15 only
(1-based). The ingestion consistency checks — relativistic-treatment compatibility, no silent
mixing of incompatible recontractions — run over the whole per-atom assignment, and atomic
mean fields and reference orbitals are solved per `(element, basis, configuration)`, so an
oxygen in TZVP and an oxygen in SVP each get the matching atomic reference.

⚠ **For an actinide that per-atom form is required, not a convenience.** No single
X2C-recontracted family covers an actinide *and* an ordinary ligand atom: the Karlsruhe
`x2c-nZVPall-2c` sets stop at Rn (Z = 86), and the Peterson `cc-pVnZ-X2C` / `cc-pwCVnZ-X2C`
sets that do reach Ac–Lr begin at K and contain no H, C, N, O or F. The assignment therefore
has to name both halves, and this is the pattern to copy:

```python
mol = kuiva.Molecule(
    atoms=[("U", (0.0, 0.0, 0.0)),
           ("O", (0.0, 0.0,  1.77)), ("O", (0.0, 0.0, -1.77))],
    basis={"U": "cc-pVTZ-X2C", "default": "x2c-TZVPall-2c"},
    charge=2, spin=0)
```

Mixing those two families across atoms is checked and passes silently: both are X2C
recontractions, so they target the same relativistic treatment. **ANO-RCC** is the one family
covering H–Cm on its own, but it is a Douglas–Kroll–Hess recontraction — putting it beside an
X2C set is allowed and **warns**, and it is on you to confirm that is what you meant.

⚠ **The honest caveat: no actinide system is validated at any tier.** The committed cross-checks
against DIRAC and OpenMolcas stop at Bi (Z = 83), and the heaviest f element in them is Dy. The
basis sets are registered, the Hamiltonian has no element cutoff, and nothing in the path is
element-specific — but "in scope" is not "tested", so an actinide number out of this code
currently has no external reference behind it.

The correction is also usable on its own, one element at a time:

```python
from pyscf import gto
from kuiva.amf import amf_correction

atom = gto.M(atom="Ne 0 0 0", basis="x2c-SVPall-2c")
corr = amf_correction(atom, method="x2camf")     # (Delta h_sf, Delta w) in the AO basis
corr.report()                                    # method, interaction, backend, magnitudes
```

### The spin-free route: turning spin–orbit coupling off entirely

`screening="none"` and `with_soc=False` are different things and are easy to confuse.
`screening="none"` drops the *two-electron* picture change and keeps the one-electron spin–orbit
operator, so the calculation is still two-component and still spin–orbit coupled — with
j-splittings 5–30 % too large. `with_soc=False` ingests no two-component Hamiltonian at all: the
correlated step runs on the spin-free X2C operator lifted to two components, and what comes out
is a scalar-relativistic spectrum whose degeneracies are `2S+1` rather than `2J+1`.

```python
scf = kuiva.ScalarSCF(mol, with_soc=False).run()      # no spin-orbit coupling anywhere
```

It is the right setting for a spin-free reference number: a term energy to compare against a
non-relativistic code, a `2S+1` term count to check a state average against before turning
coupling on, an NEVPT2 decomposition with the spin structure still intact. The front end says so
in a warning every time, because a scalar answer read as a relativistic one is the whole failure
mode. Three things are worth knowing before relying on it.

- **It costs what the two-component calculation costs.** The determinant space is still built
  over spinors, so a given active space has the full two-component determinant count, not the
  smaller α-string × β-string product a spin-free code would use. Switching spin–orbit coupling
  off makes the answer simpler, not the calculation cheaper.
- ⚠ **`Sz` is conserved here, and Kuiva does not exploit it.** With no spin–orbit coupling the
  Hamiltonian commutes with `Sz` and the determinants fall into sectors; there is no sector
  machinery, and every root is solved in the one full space.
- ⚠ **Which is exactly why the eigensolver's guess is *generic* rather than merely good.** Those
  sectors are invariant subspaces, and a Krylov method can never leave the ones its starting
  vectors lie in. A guess taken from the lowest-diagonal determinants can lie entirely inside a
  few of them — the solver then converges every residual, reports success, and returns states
  that are **not the lowest**, with no gate anywhere seeing it. Measured on a spin-free
  Dy(3+) CAS(9, 14): a 66-root solve missed 22 of the true lowest 66 and came back 6766 cm⁻¹ too
  high, in a *third* of the iterations of the correct solve — so a fast solve is evidence of
  nothing here. Generic vectors are prepended to every cold start, which makes that failure
  structurally unavailable rather than impossible: the argument is Rayleigh–Ritz interlacing,
  not a bound. ⚠ **A spectrum that changes qualitatively when you ask for a different number of
  roots is this, not physics** — and the definitive check is a dense diagonalization of the
  active-space Hamiltonian (`kuiva.ci.strings.hamiltonian_matrix`), which is cheap enough to
  reach for whenever a spin-free spectrum surprises you.

---

## Choosing and inspecting an active space

Choosing an active space means answering "are these the orbitals I meant?" — and a **spinor
cannot simply be plotted**, so Kuiva gives you two ways to answer it. Both report **degenerate
blocks, not individual spinors**, by default: a single spinor's density and populations are
basis-dependent inside a degenerate manifold, and block sums are not.

(How the space is *stated* — `character=`, the `skip_pairs` ordinal window for a second shell
of the same `l`, and `avas=` for the covalent case where no single orbital carries the
character — is in [`CASSCF`](#casscfupstream-) above. What follows is how to check it.)

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
if the spin distribution is the question. And **no atomic charge is printed in this table,
deliberately**: the Löwdin charge was measured with the wrong *sign* on ionic textbook
compounds (a negative titanium in TiCl₃, a zero cerium in CeCl₃) and a better basis does not
rescue it, so it was withdrawn from every report rather than captioned. The `atomic_charge()`
accessor remains for anyone who wants the arithmetic, as a diagnostic and never an oxidation
state. The reduced *orbital* populations are the robust half and are what this is for.

### Atomic charges that can be trusted: the atomic-reference partition

When a charge *is* wanted, ask for the **atomic-reference charges** — populations taken in an
orthonormal basis built from the orbitals of each element's spherically averaged free atom,
computed in the molecule's own basis with the same relativistic treatment. Occupied atomic
orbitals are orthogonalized first (weighted by their atomic occupations) and the atomic
virtuals are kept behind them, so bonding-region density is attributed by atomic *character*
rather than by which centre's functions happen to describe it — the failure mode that sank
the Löwdin charge. The scheme was selected by measurement against exactly that failure:
across five systems and two to four basis sets the charges keep their signs, drift by ~0.1 e
where Mulliken drifts 0.45 e and Löwdin flips sign, and a nucleus-free "ghost" basis over a
pure chloride density picks up ~0.1 e where Löwdin took 2.4.

```python
scf = kuiva.ScalarSCF(mol, atomic_reference=True).run()   # one atomic SCF per element, cached
ref = kuiva.Reference(scf).run()
ref.atomic_reference_charges()                            # prints the table, returns arrays
```

The per-element reference state is **the same default the atomic mean field uses** — the
neutral atom, and the trivalent ion on the f block — so each element has exactly one default
reference across the program. Overriding it (the same `screening_options` `configuration`
mapping the mean field takes) is honoured and **warns**: charges against a non-default
reference are not comparable with default-reference ones. `atomic_reference=True` lives on
the SCF stage because the free-atom orbitals need the integral library, which nothing
downstream has; forget it and the charges call tells you exactly that.

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
# Like memory_gb, scratch_dir has NO built-in default and no $TMPDIR/cwd fallback: any
# scratch use (a factor spill, an out-of-core decomposition, network environment paging)
# refuses until it is set here or with $KUIVA_SCRATCH — pick a
# real disk with room, never a RAM-backed tmpfs. Calculations that never touch scratch do
# not need it. setup.sh asks for it beside the memory limit.
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
| `KUIVA_SCRATCH`, `KUIVA_SCRATCH_GB` | scratch directory (no built-in default; scratch use refuses without one) and the space Kuiva may use there |
| `KUIVA_CHECKPOINT_GB` | largest checkpoint file |

Anything passed explicitly to a call overrides the environment, which overrides the
configuration file.

⚠ Kuiva reads no *scheduler* variable unless it is asked to: `$SLURM_JOB_ID` and
`$SLURM_JOB_END_TIME` are consulted only by `deadline="slurm"`/`"queue"`/`"auto"` (see
[Stopping before the queue does](#stopping-before-the-queue-does-deadline)), never on its
own initiative.

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
control at all is reported as "unverified" rather than as either verdict.

**Three BLAS libraries are controlled** — MKL, OpenBLAS and BLIS — each bound at run time to
whichever one is actually mapped into the process, including the OpenBLAS a NumPy or SciPy wheel
brings with it. So the knob, the per-stage widths and the startup measurement work the same way
on a machine without Intel's BLAS, which is most clusters. ⚠ One difference is worth knowing:
MKL's width is per-thread, while OpenBLAS's and BLIS's is process-wide, so on those a stage's
width is visible to the whole process until that stage ends. To see the numbers behind the
verdict:

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
record per matrix element: the effective Hamiltonian `H`, the three magnetic-moment components
`mu_x`, `mu_y`, `mu_z` in µ_B, and the three electric-dipole components `d_x`, `d_y`, `d_z` in
e·a₀ — all in the basis of the spin–orbit eigenstates. It is deliberately dull, because it is a
contract with an external ITO / crystal-field code, and `kuiva.read_dump` is a working parser
of it.

**Reading one back.** Because the phases in these files are arbitrary, comparing two stored
runs — or a stored run against a fresh one — has to go through the phase-invariant reduction,
so both products come back as the objects that reduction lives on:

```python
before = kuiva.PropertyMatrices.from_dump("before.props").analyse()
after  = kuiva.PropertyMatrices.from_dump("after.props").analyse()
[(b.size, b.g_values) for b in after]              # what may be compared

model = kuiva.PseudospinModel.from_file("dimer.psd")     # the same, for the export
```

⚠ **What comes back is the file, not the calculation.** The provenance, gauge origin, active
space and picture-change record are restored, because those are what make the numbers
interpretable; nothing else about the run is in the file and the object cannot be handed back
to a stage. `kuiva.read_dump` / `kuiva.read_pseudospin` remain the raw form, returning the
header verbatim as a dictionary.

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
- **The principal magnetic axes come with the g values**, in a second table beside them and on
  `Multiplet.g_axes` / `.easy_axis` / `.axiality`. ⚠ An axis is a *line*, and it is a direction
  at all only where its g value is non-degenerate: an easy-plane or isotropic block has no easy
  axis, and the vector printed for one is whatever the eigensolver picked inside the degenerate
  plane. The table says which it is on every row, and `easy_axis_is_defined()` is the test.
- **The sign of `det(g)` is reported too, and it is not a convention.** `M` is quadratic in
  `mu`, so it fixes `|det g|` and loses the sign; the sign comes back from the third-order
  invariant `det(g) = 2i·Tr(mu_x[mu_y,mu_z])/µ_B³`, which is just as unaffected by mixing
  inside the block. It is defined for a Kramers doublet only, and is reported as `?` — never
  guessed — anywhere else.
- ⚠ **A block of one state reports no g values — `nan`, never `0`.** A single state has no
  first-order magnetic moment, and "not defined here" is a different statement from "measured
  zero". The distinction is the whole of the **non-Kramers** case below.
- **The electric dipole is written by default** (`include_dipole=False` turns it off), and it
  is the **total** dipole: the electronic operator over all electrons plus, **on the diagonal
  only**, the nuclear `Σ_A Z_A (R_A − R_G)`. So a diagonal element is that state's dipole
  moment and an off-diagonal one is a transition dipole. The nuclear vector is in the header,
  so the two halves can be separated again. ⚠ The inactive electrons' share is *not* zero here
  — `r` is time **even**, unlike `L` and `S`, so a Kramers pair contributes twice its
  expectation value instead of cancelling — and it is written out in `[INACTIVE]` for the same
  reason.
- ⚠ **For a charged molecule the dipole depends on the gauge origin.** The diagonal obeys
  `d(R_G) = d(0) − q·R_G` and every invariant formed *inside* one degenerate block moves with
  it; transition elements between distinct states do not, whatever the charge. Writing such a
  file emits a `WARNING` and the header carries `molecular_charge` and
  `dipole_origin_dependence`. It is not refused — a charged lanthanide complex's transition
  dipoles are perfectly well defined, and that is what this is for.
- **The dipole has its own phase-invariant reductions, and they are the only sound comparison**:
  `Tr_block(d_i d_j)` on `Multiplet.d_tensor`, and the block-to-block **line strength**
  `S_AB = Σ_k Σ_{I∈A,J∈B} |d_k[I,J]|²` from `PropertyMatrices.line_strengths()`. ⚠ A line
  strength is **not** an oscillator strength, a radiative rate or an intensity: Kuiva writes
  operators and their invariants, and turning one into an intensity — which needs a transition
  energy, a refractive index and a convention for what is measured — belongs to the same
  external code the crystal-field analysis does.
- ⚠ **No picture change is applied to `L`, `S` or `r` by default** — see
  [Limitations](#limitations--what-not-to-trust). `property_picture_change=True` on the front
  end applies it, and it applies it to **both** the magnetic and the electric operator: there is
  deliberately no way to correct one and not the other, because nothing in such a file would say
  which half is which.
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

### ⚠ Non-Kramers ions: when the ground "doublet" is two singlets

An **integer**-spin ion — Tb(3+) ⁷F₆, Ho(3+) ⁵I₈, and the Ln single-molecule magnets this code
exists for — has no Kramers protection. Its ground doublet is two *singlets* split by a
tunnelling gap Δ, anything from 10⁻⁵ to tens of cm⁻¹. They therefore do not group at the
default 1 cm⁻¹ degeneracy tolerance, and each arrives as a block of one state — which carries
no magnetic moment, because the moment belongs to the **pair**.

So the multiplet table prints `nan` for those two blocks, and the run warns, naming Δ. Group
them explicitly to get the number you actually want:

```python
cas_matrices = kuiva.PropertyDump(cas, "tb.props").run().matrices
for m in cas_matrices.analyse(pseudo_doublet_tol_cm=50.0):
    if m.non_kramers:
        print(m.g_z, m.tunnelling_gap_cm, m.g_transverse_residual)
```

- **It is opt-in and never inferred.** Whether two nearby singlets are one tunnelling-split
  doublet or two crystal-field levels is physics their energies cannot settle, and a wrong
  grouping manufactures a plausible g out of two unrelated states.
- **`g_z` is the number to quote.** For a non-Kramers doublet the transverse components vanish
  identically (Griffith; Abragam & Bleaney), so the effective-spin description carries `g_z`
  and Δ and nothing else.
- ⚠ **`g_transverse_residual` is the check that can fail.** It is zero by symmetry for a real
  pseudo-doublet; a value comparable to `g_z` says these two states are not one, and the
  grouping request was wrong. Read it — a diagnostic whose only possible value is the one you
  assumed is not a diagnostic.
- Three near-equal singlets are not a doublet: a state consumed by one pair cannot join
  another, so nothing is quietly counted twice.
- ⚠ **Measured end to end on a real molecule** (linear FeCl₂, Fe(2+) d⁶, ⁵Δ): the non-Kramers
  ground doublet's `g_z` comes out **12.0075** against the analytic `2(Λ + g_e Σ) = 12.009`,
  its transverse g values are zero to 1e-6 — which no Kramers doublet's ever are — and bending
  the molecule to 90° splits the pair by 0.32 cm⁻¹ while the next level sits 125 cm⁻¹ away.
  That separation of scales is what an integer-spin single-molecule magnet lives on.

Both files carry a `format_version` that is bumped when the *meaning* of a stored field
changes, so a consumer can refuse rather than misinterpret.

**Checkpoints** (`CASSCF(..., checkpoint=path)`) — schema-versioned HDF5, written every
macro-iteration under an adaptive budget, with the converged one always written. They hold the
orbitals and orbital-rotation state, the active-space RDMs, the state energies and the run
metadata; CI vectors only below a size threshold, as a Davidson warm start. Four-index
integrals are never stored — a restart regenerates them, which is cheap because the expensive
atomic solves have their own persistent cache. A DMRG-CASSCF writes a second, sibling file
(`*.network.h5`) with the network state, rolling at the end of each completed sweep under the
same cadence knobs; its restart is a warm start, and its loss costs time, never correctness.

⚠ **Failure semantics are the opposite of a cache's, on purpose.** A checkpoint *write* failure
is a warning and the run continues; a *read* failure on an explicitly requested restart is an
error that propagates, because silently starting over wastes exactly the hours the file existed
to protect. A schema mismatch refuses; a changed code fingerprint only warns.

### Stopping before the queue does: `deadline=`

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

⚠ **No deadline is the default, and it is a decision rather than an omission.** A cluster
with no queue time limit is an ordinary place to run; a deadline invented for such a run
could only ever end it early for no reason. Nothing is read from the environment unless you
ask for it.

⚠ **An explicitly named source that cannot be read refuses.** `deadline="slurm"` outside a
Slurm job raises at once, naming both ways out. The alternative — quietly running with no
deadline — is the one outcome worse than either: the job is killed at the wall twelve hours
later and nothing in the output ever said the request had failed. `"auto"` is the spelling
for a script that has to run on a laptop, on an unlimited cluster and inside a queue without
being edited.

**Where the limit comes from.** Slurm's is read once, at the start: from
`$SLURM_JOB_END_TIME` (free, needs no client binaries, contacts nothing) and then from
`scontrol show job`. ⚠ Once, never in a loop — polling the controller from every
macro-iteration of every job is what makes a scheduler slow for everyone — so an allocation
*extended* after the run started is not noticed, and the run stops at the limit it was told
about. A job with no time limit (`EndTime=Unknown`) produces a deadline that never fires and
says so. Another queue system is two functions and one registry entry in
`kuiva.util.deadline`; nothing else in the program learns a scheduler's name.

⚠ **The decision is predictive, not reactive.** Stopping when the budget is already spent is
stopping too late, because the checkpoint is written *afterwards*. The run stops when

```
time left  <  longest recent macro-iteration  +  estimated checkpoint write  +  margin
```

Every term but the last is measured — the iteration times from the optimizer's own table,
the write from the measured disk bandwidth — and the margin (60 s) is printed with the rest.
The final write is then **forced** past the cadence rules, exactly as a converged one is,
because that checkpoint is the result and not insurance.

⚠ **A budget's clock starts when the stage is built, not when it is run**, so build the
stage next to its `.run()`. A queue limit is an absolute instant and cannot drift this way.
⚠ **A bare numeric string is refused** — `"60"` is sixty *minutes* to Slurm's `--time` and
would be sixty *seconds* here, and a factor of sixty in a deadline is what ends an
allocation with nothing written. Write `60`, or `"60m"`.

⚠ **The granularity is one macro-iteration.** The run stops between them and nowhere else: a
CI solve, a DMRG solve and an NEVPT2 excitation class cannot be interrupted, and if one
macro-iteration outlives the allocation no deadline can help. A run that has less than the
margin left when it starts is refused rather than started. Without `checkpoint=` a deadline
still stops the run cleanly, and says plainly that nothing was saved.

### The kill nobody announced: `signals=`

A deadline covers the case where the end is known in advance. `scancel`, a preemption, a
node draining and a `SIGTERM` at the wall are the case where it is not.

```python
cas = kuiva.CASSCF(ref, checkpoint="run.h5", deadline="slurm", signals=True, **space).run()
```

`signals=True` catches **`SIGTERM`, `SIGUSR1` and `SIGUSR2`**; a sequence names them
instead (`signals=("TERM", "INT")`). The run then stops at the next macro-iteration
boundary with its checkpoint written — the same stop, the same forced write, the same
warning as a deadline, differing only in what asked for it.

In a Slurm script, `--signal` is how you buy the lead time:

```bash
#SBATCH --signal=B:USR1@600     # SIGUSR1 ten minutes before the wall
```

⚠ The lead has to exceed one macro-iteration plus the checkpoint write, or the kill lands
first anyway. If the limit is knowable, `deadline="slurm"` is the better instrument: it
computes that reserve from what the run is actually doing instead of asking you to guess it.

Four things about it are deliberate:

- ⚠ **off by default, and never otherwise.** A library that installs signal handlers behind
  your back breaks embedding, test runners and notebooks;
- ⚠ **the handlers are a loan.** They are installed for the duration of the stage and the
  previous dispositions are restored when it ends, exception or not. Off the main thread,
  where Python cannot install a handler at all, `signals=` is **refused** rather than
  silently skipped;
- ⚠ **a second signal is not waited for.** The first is a request; the second restores what
  was there before and re-raises, so the process dies exactly as it would have without
  Kuiva in it. `SIGINT` is not in the default set for the same reason — Ctrl-C is expected
  to interrupt *now*, not at the end of a macro-iteration;
- ⚠ **the request outlives the stage that caught it.** A signal is delivered to the
  process, so an `NEVPT2`, `CASCI` or `PseudospinExport` started afterwards raises
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
| 1 | `01_scf_and_reference` | the front end end to end: scalar X2C SCF → orthonormal working basis → Kramers-paired spinors → factorized integrals, each stage checked; the SCF's convergence controls — the refusal, the stability analysis and `guess_from=`; and the two things a coupled pair of centres needs, the broken-symmetry guess and fragment localization | ~17 s |
| 2 | `02_atomic_spin_orbit` | fine structure of a free atom: the exact 2 + 4 splitting of a p shell, Landé g factors with no free parameter, what two-electron screening changes, and `⟨S²⟩` with the term assignment it supports | ~2 s |
| 3 | `03_casscf_ligand_field` | the flagship calculation: a state-averaged two-component CASSCF on TiCl₃, its active space stated as orbital character **and** built a second way by AVAS, the five Kramers doublets of the d¹ ligand field, and a term assignment that correctly refuses to name them | ~3 min |
| 4 | `04_dmrg_casscf` | the tensor-network solver: a cheap CI, a tree topology built from its entanglement, a DMRG-CASSCF reproducing the exact CI through the same orbital optimizer, the two checkpoint files of the network route, and `⟨S²⟩` with the assignment on both solver routes | ~4 min |
| 5 | `05_nevpt2` | dynamic correlation: SC-NEVPT2 on the oxygen atom, its eight-class decomposition, term energies moving towards experiment while the degeneracies survive — and the same correction from a tensor-network reference, honestly partial | ~30 s |
| 6 | `06_property_export` | the two products: the property-matrix dump and the OuluSpin pseudospin export, reaching the same g values by two independent routes | ~4 min |
| 7 | `07_checkpoint_restart` | a CASSCF checkpointed every macro-iteration, interrupted, and resumed from disk to the same energy, under a wall-clock `deadline=` | ~3 min |
| 8 | `08_slater_condon` | the extras: Slater–Condon parameters `F^k`, `G^k`, `R^k` and spin–orbit constants `ζ` of a free scandium atom, from an average-of-configuration reference | ~1 min |
| 9 | `09_basis_projection` | converging the CASSCF in a cheap basis and continuing it in the production one: `project_from=`, what the projection carries, what it measures, and the reverse direction | ~3 min |

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

  ⚠ **The same flag also corrects the electric dipole, and there is no way to correct one
  operator and not the other.** The electric operator is *even*, so its four-component matrix has
  a small-component block where the magnetic one has none, and the transformation is otherwise
  the same `X` and `R`. It is recorded in its own header field, `picture_change_on_dipole`. A file
  whose `mu` carried the correction and whose `d` did not would be a hybrid with nothing in it
  saying which half was which, which is why the two travel together.

  What it is worth is **not** what it is worth for `mu`, and the difference is physics rather than
  a defect: measured on the hydrogen halides HF → HI (`x2c-SVPall-2c`, CAS(2,2)), it moves a
  permanent dipole by **2e-06 relative** and a line strength by twice that, and — unlike the g
  factor — it does **not grow with Z**; the absolute shift in fact falls from HF to HI. The
  picture change is a near-nucleus effect and the density near a heavy nucleus is core density,
  spherical and contributing almost nothing to a dipole. It splits no degeneracy. ⚠ Those figures
  bound a *valence* dipole of a light main-group diatomic and nothing else — an `f → d` transition
  of a lanthanide, which is what this operator exists for, has not been measured.
- ⚠ **The electric dipole is length gauge only, and Kuiva computes no oscillator strengths.**
  There is no velocity-gauge second estimate, so there is no in-code number saying how converged
  a transition dipole is — judge that by enlarging the active space. And what the file carries is
  the operator and its invariants (`Tr_block(d_i d_j)`, the block-to-block line strength); f
  values, Einstein coefficients and radiative rates are the external property code's job, exactly
  as the crystal-field analysis is.
- ⚠ **DLU is measured at the state level, and it is safe for splittings but not for the
  transverse g of an axial doublet.** Against the exact decoupling through the same code, on
  d¹ and f¹ ligand fields at SA-CASSCF: splittings move by ≤0.6 cm⁻¹ and ≤0.1%, principal g
  values by ≤2e-4 relative — but the near-zero transverse g of a strongly axial doublet moves
  by ~6e-4 absolute (6% of itself), which is exactly the number a tunneling analysis reads.
  The selection warning states these bounds; check a small transverse g against
  `decoupling_options={"partition": "single"}` before quoting it. The DLU-transformed *moment
  operator* (`property_picture_change=True` with `-DLU`) remains unmeasured.
- ⚠ **Löwdin atomic charges are withdrawn from every printed report** (measured decision):
  characterized across five systems in two basis sets, the charge carries the wrong *sign* on
  three of them — a negative Ti in TiCl₃, a zero Ce in CeCl₃, a negative H in HI — and a
  better basis does not rescue it. `atomic_charge()` remains as an accessor, a diagnostic and
  never an oxidation state. Reduced *orbital* populations are robust and are what the feature
  is for. **The supported charge is the atomic-reference one** (see the populations section):
  measured stable in sign and to ~0.1 e across bases where Löwdin fails qualitatively, at the
  cost of one cached atomic SCF per element (`atomic_reference=True` on the SCF stage).
- **The conventional CI ceiling is 20–22 spinors at half filling at an 8 GB limit.** It is a
  memory bound on the *determinant count*, not on the spinor count, so it moves with the
  memory limit and is enforced before the first allocation — dilute or nearly-full spaces
  well past it run comfortably, and past 20 spinors the **state count co-decides**: a
  few-root CAS(11,22) fits an 8 GB limit while a large state average does not, and 24
  half-filled spinors refuse at any root count. The hard limit is 64 spinors (a single
  64-bit occupation mask). Beyond the ceiling, the tensor-network solver takes over.
- **The integral factorization is memory-bound, and the memory plan picks the route.**
  The stored route materializes the conventional two-electron integral array, which grows as
  the fourth power of the basis and is reserved against the memory limit before the SCF —
  though only until the factors replace it, since it is released there. `fitting="auto"` (the
  default) switches to `cholesky-direct` — which never forms that array, see above — when
  the stored plan is the larger one and the two-electron phase is what peaks; below the size
  at which the array dominates, the two plans are the same after the release and the stored
  route stands. The direct route is *not* faster where both fit (measured within a few per
  cent of the stored route from ~160 basis functions up, and up to ~46% slower when its batch
  cache is squeezed at very large sizes); what it buys is that the calculation starts at all.
  The three-index factors it produces are then the next term — `factors="streamed"` keeps
  even those out of memory, at the cost of repeated passes over a scratch file. ⚠ Note also that the
  scalar SCF is PySCF's, and it makes its own in-core/direct decision within the memory it is
  given: the direct route removes Kuiva's copy of the array, not necessarily every copy.
- **Kramers degeneracy in the general two-component CI emerges numerically, not by
  construction** — that path is the default and every reference in this project is one of its
  results. In practice the measured splitting is far below the 1e-8–1e-6 Eh band reserved for a
  genuine numerical splitting, but it is not zero by symmetry. The **tensor-network** solver's
  figure is larger and for a different reason: a sweep converges its roots separately, leaving
  of order 1e-9 Eh between paired states even where nothing is truncated. Both are far below
  anything physical; neither is a bound on the other.
- **The Kramers-restricted (time-reversal-adapted) CI covers odd electron counts only, and is
  worth a factor of two only above two Kramers pairs.** `solver_options=dict(
  kramers="restricted")` on a CASSCF keeps the eigensolver's expansion subspace closed under
  time reversal, so one stored direction spans a whole Kramers pair: measured 1.8–1.9× less CPU
  from three averaged pairs upward, and ⚠ **8% *slower* at two pairs**, where a block Davidson
  adds one expansion direction per iteration instead of two. It refuses an even electron count
  (a different theorem, giving a real Hamiltonian rather than a halved subspace — not
  implemented) and an odd state count, and it refuses integrals whose orbitals are not
  Kramers-paired. ⚠ It does **not** raise the active-space ceiling: the sigma-vector residency
  that sets the ceiling is a whole-space gather in either mode.
- **Point-group symmetry is abelian double groups only, opt-in, and does not make a state
  average safe.** `point_group=` labels the orbitals and enables per-irrep state selection and a
  symmetry-preserving orbital optimization; without it, states are the lowest *n* roots exactly
  as before. Three limits are real. The operations are tested in the frame the geometry is given
  in — the molecule is never reoriented, because that would move the gauge origin — so a
  symmetry axis that is not `z` is not used. Groups whose *double* group is non-abelian (`C2v`,
  `D2`, `D2h`) are reduced to the largest subgroup that has one-dimensional fermion irreps —
  and where that happens the converged states are also *classified* by the full double group's
  irreps, which is a labelling of the finished spectrum and changes nothing about how it was
  computed. And because the group used in the mathematics can be smaller than the molecule's
  real one, two members of a
  physically degenerate manifold can carry different labels: **a per-irrep count can split a
  manifold exactly as a plain count can**, so the state-average boundary diagnostic stays what
  catches it. Non-abelian symmetry adaptation is out of scope, and labelling states by
  non-abelian irreps is not implemented.
- **NEVPT2 is strongly contracted only.** FIC and
  quasi-degenerate variants are not implemented and are not planned — the artificial multiplet
  splitting they would cure is four orders of magnitude below the size at which a splitting
  means different physics. ⚠ A tensor-network reference reaches NEVPT2 through the
  network-backed contraction provider with **six of the eight classes**: the primed
  single-external ones (`Sr`, `Si`) are not served yet, so that `E2` is a **partial** sum —
  skipped with a warning, marked incomplete on the result, printed as PARTIAL — and is not
  comparable with a complete NEVPT2.
- **One node, shared memory.** There is no MPI and no distributed tensor layer. Memory, not
  core count, is the scaling limit.
- ⚠ **A term label is an inference and is printed as one.** `assign()` derives `^{2S+1}L_J`
  from three measurements — the block dimension, `<S^2>` and the isotropic g through the
  inverted Landé formula — and every row prints the evidence and a fit residual beside the
  label. It is a separate report, never a column of the state table, and never written to a
  stored file. A block whose evidence does not add up is labelled `?`, which is the normal
  outcome for the crystal-field levels of a complex: those are not `2J+1` manifolds, so the
  inversion has nothing to invert. Do not quote a label without the residual next to it.
  `<S^2>` itself is a measurement and is trustworthy — but only per **degenerate block**
  (a single state's value inside one is basis-dependent). Both calls work on either solver
  route; the tensor-network side contracts the same quantities through per-root densities
  and network transition densities.
- **Magnetic properties themselves are out of scope.** Kuiva writes the operator matrices; the
  ITO / Stevens / crystal-field decomposition is an external code's job.
- **Four-component methods are out of scope.** Four-component machinery exists inside the code
  as an *ingredient* of the two-electron picture change, not as a method path.
- **An unrestricted reference gives spinors that are not Kramers paired**, so an active space
  cannot be taken as a contiguous spinor range there.
- **Absolute total energies are bounded by the basis, not by the code's defaults.** The
  Cholesky threshold default was decided on *relative* energies, where the factorization error
  largely cancels: at the default `1e-8` a 20-state Bi spectrum is right to 1.5e-04 cm⁻¹ while
  its absolute total is off by 3.7e-07 Eh — a factor of ~530. The factorization is a knob
  rather than a floor, and tightening it to `1e-10` brings the absolute total inside 1e-8 Eh
  on every system measured, for ~25% more auxiliary vectors. ⚠ That is worth doing only when
  two totals are compared *in the same basis*: walking x2c-SVPall-2c → TZVP → QZVP moves an
  absolute total by 1.7e-01 to 7.6e-01 Eh and then by a further 5.5e-03 to 9.9e-02 Eh, still
  unconverged at QZVP — so for any total energy actually being quoted the threshold is the
  smallest error present by six to eight orders of magnitude. ⚠ A Kuiva total also omits the mean-field `ΔG` (see above), so it is
  not directly comparable with a four-component total in any case.

---

## Versioning

**Version 0.33.0.** The number is `MAJOR.MINOR.PATCH` and reads as usual:

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
import kuiva; kuiva.__version__          # '0.33.0'
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
- **OpenBLAS** and **BLIS** — the two other BLAS libraries Kuiva sets a thread width on, through
  `openblas_set_num_threads` / `openblas_get_num_threads` / `openblas_get_config` and
  `bli_thread_set_num_threads` / `bli_thread_get_num_threads`. Z. Xianyi, W. Qian, Z. Chothia,
  _OpenBLAS_, https://www.openblas.net/; F. G. Van Zee, R. A. van de Geijn, _BLIS: A Framework
  for Rapidly Instantiating BLAS Functionality_, _ACM Trans. Math. Softw._ **41**, 14 (2015),
  DOI:10.1145/2764454.
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
- **Finite (Gaussian) nuclear charge distribution**, `nuclear_model="gaussian"`. L. Visscher,
  K. G. Dyall, _At. Data Nucl. Data Tables_ **67**, 207 (1997), DOI:10.1006/adnd.1997.0751 —
  the parametrization of the rms radius from the isotope mass, which is what PySCF's
  `dyall_nuc_mod` implements and what Kuiva selects.
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
- **Common oxidation states** (the curated table behind `configuration=`): N. N. Greenwood,
  A. Earnshaw, _Chemistry of the Elements_, 2nd ed., Butterworth-Heinemann (1997). Atomic
  ground-state configurations follow the NIST Atomic Spectra Database via PySCF's aufbau
  tables.

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
- **Projecting molecular orbitals from one basis set onto another**, and the Fock/density-matrix
  alternative to it (`OVPROJECTION` and `FOPPROJECTION` in Q-Chem's terms). R. P. Steele,
  R. A. DiStasio Jr., Y. Shao, J. Kong, M. Head-Gordon, _J. Chem. Phys._ **125**, 074108 (2006),
  DOI:10.1063/1.2234371; projected starting vectors from a smaller basis as the standard SCF
  guess, J. Almlöf, K. Fægri, K. Korsell, _J. Comput. Chem._ **3**, 385 (1982),
  DOI:10.1002/jcc.540030314.
- **The same operation for the active orbitals of a CASSCF, as a production workflow** —
  OpenMolcas' `EXPBAS`. I. Fdez. Galván _et al._, _J. Chem. Theory Comput._ **15**, 5925 (2019),
  DOI:10.1021/acs.jctc.9b00532; F. Aquilante _et al._, _J. Chem. Phys._ **152**, 214117 (2020),
  DOI:10.1063/5.0004835.
- **Least-squares optimality of symmetric orthonormalization**, which is why the projected set is
  repaired that way. B. C. Carlson, J. M. Keller, _Phys. Rev._ **105**, 102 (1957),
  DOI:10.1103/PhysRev.105.102.
- **Principal angles between orbital subspaces** ("corresponding orbitals"), the invariant the
  projection is judged by. A. T. Amos, G. G. Hall, _Proc. R. Soc. London A_ **263**, 483 (1961),
  DOI:10.1098/rspa.1961.0175.

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
  J. Olsen, L. Visscher, _J. Chem. Phys._ **119**, 2963 (2003), DOI:10.1063/1.1590636;
  T. Saue, H. J. Aa. Jensen, _J. Chem. Phys._ **111**, 6211 (1999), DOI:10.1063/1.479958.
- **Self-dual (quaternion) Hermitian eigenvalue structure**, which is what the
  Kramers-restricted eigensolver's subspace problem is. A. Bunse-Gerstner, R. Byers,
  V. Mehrmann, _SIAM J. Matrix Anal. Appl._ **10**, 419 (1989), DOI:10.1137/0610030.
- **Corresponding orbitals** (the standard way to pair two non-orthogonal orbital sets).
  A. T. Amos, G. G. Hall, _Proc. R. Soc. London A_ **263**, 483 (1961), DOI:10.1098/rspa.1961.0175.
- **UHF natural orbitals as an active-space guess.** P. Pulay, T. P. Hamilton, _J. Chem. Phys._
  **88**, 4926 (1988), DOI:10.1063/1.454704.

### Point-group and double-group symmetry

- **Double-group character tables**, stored in `kuiva/symm/groups.py` and printed by every
  symmetric run. G. F. Koster, J. O. Dimmock, R. G. Wheeler, H. Statz, _Properties of the
  Thirty-Two Point Groups_, MIT Press (1963); S. L. Altmann, P. Herzig, _Point-Group Theory
  Tables_, Clarendon Press (1994).
- **The spin-1/2 representation and the `E`/`Ebar` distinction** that makes a double group a
  double group. E. P. Wigner, _Group Theory and its Application to the Quantum Mechanics of
  Atomic Spectra_, Academic Press (1959), ch. 15; H. A. Bethe, _Ann. Phys._ **3**, 133 (1929),
  DOI:10.1002/andp.19293950202.
- **Abelian double groups as the working symmetry of a two- or four-component molecular code**,
  which is the design this follows. T. Saue, H. J. Aa. Jensen, _J. Chem. Phys._ **111**, 6211
  (1999), DOI:10.1063/1.479958; L. Visscher, _Chem. Phys. Lett._ **253**, 20 (1996),
  DOI:10.1016/0009-2614(96)00234-5.
- **Rotation matrices for real spherical harmonics by recursion**, which is how the full
  group's operators are built on the AO basis. J. Ivanic, K. Ruedenberg, _J. Phys. Chem._
  **100**, 6342 (1996), DOI:10.1021/jp953350u, and the erratum, _J. Phys. Chem. A_ **102**,
  9099 (1998), DOI:10.1021/jp9833350.
- **Character tables from the class-sum matrices**, which is why the full double-group tables
  printed by a classifying run are computed rather than transcribed. W. Burnside, _Theory of
  Groups of Finite Order_, 2nd ed., Cambridge University Press (1911), ch. XV; J. D. Dixon,
  _Numer. Math._ **10**, 446 (1967), DOI:10.1007/BF02162876.
- **Irrep naming**, for the single-valued rows. R. S. Mulliken, _J. Chem. Phys._ **23**, 1997
  (1955), DOI:10.1063/1.1740655.
- **Projection of a reducible representation onto characters**, which is what classifies a
  converged degenerate block. M. Tinkham, _Group Theory and Quantum Mechanics_, McGraw-Hill
  (1964), ch. 3.
- **The Fock-space image of a one-body unitary**, applied here as a product of two-mode
  rotations so that a symmetry operation can be applied to a CI vector exactly.
  D. J. Thouless, _Nucl. Phys._ **21**, 225 (1960), DOI:10.1016/0029-5582(60)90048-1; the
  adjacent-pair elimination is the Givens QR of G. H. Golub, C. F. Van Loan, _Matrix
  Computations_, 4th ed., Johns Hopkins University Press (2013), section 5.2.

### Integral factorization and transformation

- **Cholesky decomposition of the two-electron integral matrix.** N. H. F. Beebe, J. Linderberg,
  _Int. J. Quantum Chem._ **12**, 683–705 (1977), DOI:10.1002/qua.560120408.
- **Pivoting, error bounds and modern practice**, and the integral-direct formulation in which
  only the selected columns are ever evaluated (`fitting="cholesky-direct"`). H. Koch,
  A. Sánchez de Merás, T. B. Pedersen, _J. Chem. Phys._ **118**, 9481 (2003),
  DOI:10.1063/1.1578621; F. Aquilante, T. B. Pedersen, R. Lindh, _J. Chem. Phys._ **126**,
  194106 (2007), DOI:10.1063/1.2736701; F. Aquilante _et al._, in _Linear-Scaling Techniques in
  Computational Chemistry and Physics_, Springer (2011), pp. 301–343,
  DOI:10.1007/978-90-481-2853-2_13.
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
- **AVAS — atomic valence active space**, the projection-and-rotation route to an active space
  where no canonical orbital carries the target character on its own. E. R. Sayfutyarova,
  Q. Sun, G. K.-L. Chan, G. Knizia, _J. Chem. Theory Comput._ **13**, 4063 (2017),
  DOI:10.1021/acs.jctc.7b00128. ⚠ `kuiva` follows the projection, the occupied/virtual
  separation and the eigenvalue threshold, but projects onto its **own free-atom reference
  orbitals** rather than a minimal MINAO basis — so the orbitals selected agree while the
  eigenvalues are not numerically comparable with another program's AVAS.
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
  (2003), DOI:10.1103/PhysRevB.67.125114. The truncation perturbation (density-matrix
  noise / subspace expansion, implemented as the deterministic expansion): S. R. White,
  _J. Chem. Phys._ **122**, 084108 (2005), DOI:10.1063/1.1854132; C. Hubig, I. P. McCulloch,
  U. Schollwöck, F. A. Wolf, _Phys. Rev. B_ **91**, 155115 (2015),
  DOI:10.1103/PhysRevB.91.155115. Energy extrapolation in the discarded weight:
  G. K.-L. Chan, M. Head-Gordon, _J. Chem. Phys._ **116**, 4462 (2002), DOI:10.1063/1.1449459;
  R. Olivares-Amaya, W. Hu, N. Nakatani, S. Sharma, J. Yang, G. K.-L. Chan, _J. Chem. Phys._
  **142**, 034102 (2015), DOI:10.1063/1.4905329. Automatic structural optimization of tree tensor
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
- **NEVPT2 on a DMRG reference.** S. Guo, M. A. Watson, W. Hu, Q. Sun, G. K.-L. Chan,
  _J. Chem. Theory Comput._ **12**, 1583 (2016), DOI:10.1021/acs.jctc.6b00118 — the
  precedent; `kuiva`'s network provider serves the same primitives through applied-string
  Gram contractions instead of stored higher densities.
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
- **Landé g factors and free-ion multiplets** — and the Russell–Saunders counting the term
  assignment inverts them with. R. D. Cowan, _The Theory of Atomic Structure and Spectra_,
  University of California Press (1981), ch. 4 and 11.
- **The non-Kramers doublet**, its tunnelling splitting and the effective-spin form in which
  the transverse g components vanish identically. J. S. Griffith, _Phys. Rev._ **132**, 316
  (1963), DOI:10.1103/PhysRev.132.316; Abragam & Bleaney (1970), ch. 3.11 and 18.3.
- **`⟨S²⟩` as a one- plus two-body operator** and its use as a spin-contamination diagnostic.
  A. Szabo, N. S. Ostlund, _Modern Quantum Chemistry_, McGraw-Hill (1989), ch. 2.5 and 3.8.

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
