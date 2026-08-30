# Kuiva documentation

Kuiva performs **relativistic multireference calculations** on strongly correlated, strongly
relativistic systems — 5d transition-metal, lanthanide and actinide complexes, including
single-molecule-magnet targets. These pages are the manual: how to install and configure the
program, how to run calculations from the simple to the difficult, what every method computes
— down to the working equations — and what every option changes.

The repository-root `README.md` is only the front door; everything of substance is here.

## How the documentation is organized

- **[`guide/`](#getting-started)** — task-ordered: installation, configuration, clusters, a first
  calculation, and the harder real workflows. Start here to *run* something.
- **[`methods/`](#the-methods)** — organized by physics: what equation each stage solves,
  exactly as implemented, with citations. Start here to *understand* (or to compare against
  another program) what a number means.
- **[`reference/`](#the-reference)** — organized by API surface: one page per stage class,
  every option listed concisely and elaborated, the file formats, the environment variables.
- **[`notation.md`](notation.md)** — the conventions every equation in these pages assumes:
  units, index conventions, the spinor basis, integral and density-matrix conventions. Read
  it once; every method page leans on it.
- **[`references.md`](references.md)** — the bibliography. Every method, algorithm, basis
  set and library Kuiva implements, follows or depends on, as one numbered list. The policy
  is to over-cite rather than under-cite: if a published idea is used, it is cited.

Every user-facing option is documented twice, on purpose: one row in its stage's
quick-reference table (for looking it up), and a fuller discussion of its consequences where
they belong, usually on a method page (for deciding whether to use it). The two link to each
other.

## Contents

Pages marked *in preparation* exist in the plan but are not yet written; they gain links as
they land.

### Getting started

| page | contents |
|---|---|
| [installation](guide/installation.md) | requirements, virtual environment, `pip install`, `setup.sh`, the optional compiled backend |
| [configuration](guide/configuration.md) | `defaults.conf`, environment variables, the thread budget, the memory limit, the scratch directory |
| [clusters](guide/clusters.md) | batch scripts, wall-clock deadlines, signal handling, checkpointing and restart, caches on shared filesystems |
| [first calculation](guide/first-calculation.md) | one complete calculation, start to finish, every line explained |
| [workflows](guide/workflows.md) | state-average design, double shells, broken-symmetry references, carrying orbitals between basis sets, embedding, spin-free runs, NEVPT2, the tensor-network route |

### The methods

| page | contents |
|---|---|
| [overview](methods/overview.md) | the pipeline, stage by stage; the physical strategy; which page covers what |
| [x2c](methods/x2c.md) | the exact two-component decoupling, working equations, local (DLU) decoupling, nuclear charge models |
| [soc](methods/soc.md) | the spinor basis, Kramers pairs, two-electron spin–orbit coupling by atomic mean field (X2CAMF), the molecular mean field |
| [scf-reference](methods/scf-reference.md) | the scalar-relativistic SCF: RHF/ROHF/UHF, broken symmetry, warm starts, average of configuration |
| [integrals](methods/integrals.md) | the orthonormal working basis, linear-dependence removal, Cholesky decomposition and density fitting, basis projection |
| [active-spaces](methods/active-spaces.md) | selection by orbital character, ordinal windows, AVAS, the cheap CI, fragment localization |
| [ci](methods/ci.md) | complex determinant CI, the sigma vector, the Davidson solver, the Kramers-restricted mode |
| [casscf](methods/casscf.md) | the state-averaged orbital optimization, the gradient, state selection and its diagnostics, adaptive solvers |
| [dmrg](methods/dmrg.md) | the tree tensor network solver: TTNO, sweeps, truncation, densities, local multiplets |
| [nevpt2](methods/nevpt2.md) | strongly contracted NEVPT2 in spinor second quantization: the Dyall Hamiltonian, the eight excitation classes, degenerate manifolds |
| [symmetry](methods/symmetry.md) | abelian double groups, orbital and state labels, classification by the full double group |
| [properties](methods/properties.md) | magnetic moments and g values, electric dipoles, phase-invariant analysis, populations, term assignment, the pseudospin mapping |

### The reference

| page | contents |
|---|---|
| [stages](reference/stages/README.md) | the shared stage contract, plus one page per class: [Molecule](reference/stages/Molecule.md) (and `Environment`, `CustomBasis`), [ScalarSCF](reference/stages/ScalarSCF.md), [Reference](reference/stages/Reference.md), [CheapCI](reference/stages/CheapCI.md), [CASSCF](reference/stages/CASSCF.md), [CASCI](reference/stages/CASCI.md), [NEVPT2](reference/stages/NEVPT2.md), [PropertyDump](reference/stages/PropertyDump.md), [PseudospinExport](reference/stages/PseudospinExport.md) |
| [api](reference/api.md) | the function layer beneath the stage classes, the top-level readers, and the extras |
| [files](reference/files.md) | the stored products as contracts: the checkpoint files, the property dump, the pseudospin export, molden files, the Slater-Condon file |
| [options index](reference/options-index.md) | one flat table from every option name to where it is documented |
| [environment variables](reference/environment-variables.md) | every `KUIVA_*` variable in one place |

### Shared vocabulary, and what not to trust

| page | contents |
|---|---|
| [notation](notation.md) | units, physical constants, index conventions, the spinor basis, second quantization, density matrices, degenerate blocks and phases |
| [limitations](limitations.md) | **read before trusting a number**: every approximation of measured size, every ceiling, and every quantity that must be read with its context |
| [references](references.md) | the numbered bibliography |

## Conventions in these pages

- **Citations** are bracketed numbers — [[16]](references.md#r16) — linking into
  [references.md](references.md). A number identifies one work permanently; the list only
  ever grows.
- **⚠** marks a trap: a way of getting a plausible-looking wrong result with no error
  message. These are worth reading even when skimming.
- **Equations** state what the code computes, in the conventions of
  [notation.md](notation.md). Where Kuiva departs from the formulation of a cited paper, the
  page says so at that point.
- Program output shown in code blocks is reproduced verbatim.
