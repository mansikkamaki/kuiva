# The stage classes

The primary user surface: one class per pipeline stage, imported from the top level
(`kuiva.ScalarSCF`, `kuiva.CASSCF`, …). This directory holds one reference page per class —
a quick option table first, elaboration below — plus a page for the input objects the
pipeline starts from.

| page | class | built from | what it does |
|---|---|---|---|
| [Molecule](Molecule.md) | `Molecule`, `Environment`, `CustomBasis` | — | the input objects: geometry, basis assignment, point charges, custom basis sets, ghost atoms |
| [ScalarSCF](ScalarSCF.md) | `ScalarSCF` | a `Molecule` | scalar-relativistic X2C SCF, integral ingestion, the two-component spin–orbit Hamiltonian |
| [Reference](Reference.md) | `Reference` | `ScalarSCF` | orthonormal working basis, Kramers-paired spinor guess, factorized two-electron integrals |
| [CheapCI](CheapCI.md) | `CheapCI` *(optional)* | `Reference` | cheap selected-CI pre-optimization: physical active orbitals, entanglement data |
| [CASSCF](CASSCF.md) | `CASSCF` | `Reference` or `CheapCI` | state-averaged two-component CASSCF, `solver="ci"` or `"dmrg"` |
| [CASCI](CASCI.md) | `CASCI` *(optional)* | `Reference`, `CheapCI` or `CASSCF` | a full CI at **fixed** orbitals — the scan primitive |
| [NEVPT2](NEVPT2.md) | `NEVPT2` *(optional)* | `CASSCF` or `CASCI` | SC-NEVPT2, per state, by excitation class |
| [PropertyDump](PropertyDump.md) | `PropertyDump` | `CASSCF`, `CASCI` or `NEVPT2` | the property-matrix file: `H`, `mu_x/y/z`, `d_x/y/z` |
| [PseudospinExport](PseudospinExport.md) | `PseudospinExport` | `CASSCF` | local multiplets, `H_eff` and moments on a pseudospin product basis |

## The contract every stage obeys

Learn one, guess the rest:

- **The constructor takes the finished upstream stage** plus keyword options, and validates
  everything it can immediately — by name against the underlying driver's signature, so a
  misspelled option, an impossible active space or a missing prerequisite fails at
  construction, not an hour into the run. An upstream stage that has not been `.run()` is
  refused.
- **`.run()` is the only expensive call.** It executes the stage, stores the results as
  plain attributes and returns `self`, so `cas = CASSCF(ref, ...).run()` reads linearly.
  It is idempotent: a second call returns the same finished object without recomputing.
- **`.summary()`** returns a short plain-text block of the headline results.
- **Results are plain attributes**, and the low-level result objects stay reachable
  (`.data`, `.reference`, `.outcome`, `.result`).
- Options that name times or signals (`deadline=`, `signals=`) are **resolved at
  construction** — `deadline="slurm"` outside a Slurm job, or `signals=` off the main
  thread, fails immediately rather than after the first hour.

The stage layer is a thin wrapper over `kuiva.interface.api` and the module drivers, which
remain public, unchanged and usable directly — `spinor_reference`, `casscf`, `sc_nevpt2`,
`optimize_orbitals` and the rest are where you go to drive one piece of the pipeline by
hand. The `kuiva` top level itself stays thin on purpose: besides `Molecule` and the stage
classes it carries only the *read* counterparts of what the stages write — `read_dump`,
`read_pseudospin`, `read_checkpoint`, `read_nevpt2_checkpoint`, `PropertyMatrices`,
`PseudospinModel` — because reading a stored product back is how two calculations get
compared at all.

Cluster-facing behaviour shared by the long-running stages — `checkpoint=`/`restart=`,
`deadline=`, `signals=`, `from_checkpoint` — is described once, in
[Running on clusters](../../guide/clusters.md), and only stage-specific details are repeated
on the pages here.
