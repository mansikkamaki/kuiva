# A first calculation

A complete calculation, start to finish, every line explained: planar TiCl₃ — Ti(III), one
3d electron in a D₃ₕ ligand field. It takes a couple of minutes on a laptop. This page
assumes [installation](installation.md) is done and a memory limit is
[configured](configuration.md).

A calculation is a short **Python script**, not a text input file: one object per pipeline
stage, each built from the finished stage before it.

```python
import kuiva
from kuiva.util.logging import add_file_handler

add_file_handler("ticl3.out")          # the run's output, to a file as well as the terminal

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

## The pipeline at a glance

| stage | built from | what it does |
|---|---|---|
| `ScalarSCF` | a `Molecule` | scalar-relativistic X2C SCF (RHF/ROHF/UHF), integral ingestion, the two-component spin–orbit Hamiltonian |
| `Reference` | `ScalarSCF` | orthonormal working basis, Kramers-paired spinor guess, factorized two-electron integrals |
| `CheapCI` *(optional)* | `Reference` | cheap selected-CI pre-optimization: physical active orbitals, entanglement data |
| `CASSCF` | `Reference` or `CheapCI` | state-averaged two-component CASSCF [[93]](../references.md#r93)[[106]](../references.md#r106), `solver="ci"` or `"dmrg"` |
| `CASCI` *(optional)* | `Reference`, `CheapCI` or `CASSCF` | a full CI at **fixed** orbitals: a spectrum without a second orbital optimization |
| `NEVPT2` *(optional)* | `CASSCF` or `CASCI` | strongly contracted NEVPT2 [[144]](../references.md#r144), per state, by excitation class |
| `PropertyDump` | `CASSCF`, `CASCI` or `NEVPT2` | the property-matrix file: `H`, the magnetic moments, the electric dipoles |
| `PseudospinExport` | `CASSCF` | local multiplets, effective Hamiltonian and moments on a pseudospin product basis |

**One contract, obeyed by all of them** — learn one, guess the rest:

- the **constructor takes the finished upstream stage** plus keyword options, and validates
  everything it can immediately. A misspelled option, an impossible active space or a
  missing prerequisite fails at construction, not an hour into the run;
- **`.run()` is the only expensive call.** It executes the stage, stores the results as
  plain attributes and returns `self`, so `cas = CASSCF(ref, ...).run()` reads linearly.
  Calling it twice returns the same object without recomputing;
- **`.summary()`** returns a short plain-text block of the headline results;
- results are plain attributes, and the low-level objects stay reachable (`.data`,
  `.reference`, `.outcome`, `.result`). The stage layer is a thin wrapper over
  `kuiva.interface.api` and the module drivers, which remain public and usable directly for
  driving one piece of the pipeline by hand.

## What each line does

**The molecule.** Geometry is in Angstrom (the default — see
[notation](../notation.md#units-and-physical-constants)), the basis is the Karlsruhe
`x2c-SVPall-2c` set [[33]](../references.md#r33), an X2C recontraction with the `-2c`
variant meant for two-component work. `spin=1` is the number of unpaired electrons.
All-electron bases only: an ECP basis is refused, because X2C has no meaning with one.

**`ScalarSCF`.** A *scalar-relativistic* (spin-free) X2C SCF: the orbitals are real and easy
to converge, and spin–orbit coupling is deliberately **not** in them — it enters later, when
the two-component wavefunction is built and optimized. Alongside the SCF, this stage ingests
the integrals and builds the full two-component X2C Hamiltonian with the two-electron
spin–orbit screening (X2CAMF [[16]](../references.md#r16)) applied by default — that default
costs one cached four-component atomic solve per unique element. An SCF that runs out of
cycles **refuses** rather than handing on whichever iterate the budget stopped at:
everything downstream is built on those orbitals, and "the CASSCF re-optimizes them" is a
hope, not a property (`allow_unconverged_scf=True` proceeds deliberately and warns).

**`Reference`.** The orthonormal working basis (linear dependence removed at a stated,
reported threshold), the Kramers-paired spinor expansion of the scalar orbitals, and the
Cholesky factorization of the two-electron integrals [[74]](../references.md#r74)
[[78]](../references.md#r78) — Cholesky by default because its error is a threshold you set,
where a density-fitting error is not bounded at all.

**`CASSCF`.** The multireference step: a state-averaged, fully complex two-component CASSCF
whose CI roots *are* the spin–orbit eigenstates [[106]](../references.md#r106)
[[107]](../references.md#r107) — there is no separate spin–orbit mixing step afterwards.
Three of its arguments are decisions you will make in every real run, and they are the
reason the API asks for them explicitly.

## The three decisions

**1. The active space is stated as orbital character, not as indices.** *"The five lowest
Kramers pairs of d character on the titanium"* is a definition another program can
reproduce; *"spinors 68 to 77"* is not, and it silently follows the basis set around. A
principal quantum number is refused on purpose: PySCF's principal-quantum AO labels count
shells *within the basis*, which is not the same thing. ⚠ One trap to know from the start:
`character=` takes the **lowest** qualifying pairs, which is the valence shell only when
nothing of that angular momentum is filled beneath it — true for 3d and 4f, false for every
p shell above the second row. The fix (an ordinal window) and the covalent case (AVAS) are
in [workflows](workflows.md#active-spaces-beyond-the-simple-case).

**2. Which states the orbitals are optimized for.** The state average here is the ground
Kramers doublet, because that is the state whose orbitals are wanted. Averaging over
everything would optimize the orbitals for an average nobody asked about.

**3. Where the state average stops.** A state-averaged CASSCF is exactly as symmetric as the
set it averages over. A count that stops *inside* a near-degenerate manifold makes the
averaged density non-invariant, the Fock operator built from it splits the shell, the
orbitals optimize on that broken density, and the selection keeps cutting the same way — the
result is self-reinforcing and entirely plausible. The only evidence is a root the average
does *not* use, so Kuiva solves a few extra, discards them, and reports the boundary gap —
at the starting orbitals as well as at the converged ones, since the starting one is what
says whether the trajectory was safe. Below 50 cm⁻¹ it warns. The diagnostic never kills a
run: a failure to measure the gap is a warning and no report, which is a weaker statement
than a clean one.

A clean gap is necessary and not sufficient, so the converged boundary report also states
the **spin non-invariance** of the averaged density — whether the ensemble is one the
symmetry leaves invariant at all. Near zero (every term-complete or full-space average)
means safe by symmetry. Large means the average *leans* on the spin–orbit structure and is
protected only by its boundary gap and its starting orbitals. The full checklist for
designing a state average — including the measured failure case no diagnostic can see —
is in [workflows](workflows.md#designing-a-state-average); read it before running anything
harder than a ground doublet, because the state average is the single most common way to get
a plausible wrong answer out of this program.

## Reading the output

Everything the run prints goes through one output grammar, so the output file reads like a
conventional quantum-chemistry output: a banner (version, threads, BLAS verdict, kernel
backend), sections per stage, label/value blocks and fixed-width iteration tables, ASCII
only. `WARNING` and `ERROR` lines are prefixed `*** WARNING [subsystem]` — they deliberately
break the visual flow and are greppable; read every one. Verbosity is per subsystem
(`set_verbosity("DEBUG")` for per-micro-iteration detail).

Things worth looking at in this run:

- the SCF block: convergence, and the `<S^2>` of the ROHF reference;
- the CASSCF iteration table: energy, gradient norm, and the boundary-gap lines at the start
  and at convergence;
- `cas.summary()` and `cas.energies` — the two states are a Kramers doublet, degenerate to
  well below any physical tolerance;
- `cas.spin_analysis().report()` — `<S^2>` per degenerate block; expect ≈ 0.75 with a small
  spin–orbit excess, which is a *measurement* of the spin mixing, not an error;
- `cas.assign()` — a term-label *offer* with its evidence and fit residual. For a free ion
  it names the term; for the crystal-field levels of a complex, `?` is the normal and
  correct outcome.

## The product

`PropertyDump` writes the deliverable: a plain-text file with a versioned header, the full
Hamiltonian provenance, and the effective Hamiltonian, magnetic-moment and electric-dipole
matrices in the basis of the spin–orbit eigenstates — the contract with an external
crystal-field / ITO analysis code. ⚠ The matrix *phases* in it are arbitrary and are not
canonicalized, so two runs are compared through phase-invariant reductions only
[[179]](../references.md#r179):

```python
m = kuiva.PropertyMatrices.from_dump("ticl3.props").analyse()
[(b.size, b.g_values) for b in m]      # degeneracy pattern and principal g values
```

For this molecule the ground doublet's g values are the physical output — and for a free ion
the same reduction has an *analytic* target (the Landé factor
[[181]](../references.md#r181)), independent of every program involved, which is the
cheapest external check there is.

## Next

[Workflows](workflows.md) — state-average design, double shells, broken symmetry, carrying
orbitals between basis sets, embedding — and, for long jobs,
[running on clusters](clusters.md).
