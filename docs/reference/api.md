# The function layer

The stage classes are a thin layer over `kuiva.interface.api` (`api` below) and the module
drivers, all of which stay public, unchanged and usable directly. Reaching for one is a
deliberate step off the documented path; each is still bound by the same memory limit, the
same provenance records and the same state-averaging gate as a stage — it is the same code
underneath. Docstrings are the authority for full signatures; this page is the map.

## The top level

`kuiva` itself stays thin on purpose. Besides `Molecule`, `Environment`, `CustomBasis` and
the stage classes it carries only the **read counterparts** of what the stages write —
because reading a stored product back is how two calculations get compared at all:

| name | what it reads |
|---|---|
| `kuiva.read_dump(path)` | a property dump, raw: matrices plus the header verbatim as a dict |
| `kuiva.PropertyMatrices.from_dump(path)` | the same, as the object the phase-invariant `.analyse()` lives on |
| `kuiva.read_pseudospin(path)` / `kuiva.PseudospinModel.from_file(path)` | the pseudospin export, raw / as the model object |
| `kuiva.read_checkpoint(path)` | a CASSCF checkpoint's contents |
| `kuiva.read_nevpt2_checkpoint(path)` | the per-`(state, class)` NEVPT2 table |

⚠ What comes back from a stored product is the **file, not the calculation**: provenance,
gauge origin, active space and picture-change record are restored — what makes the numbers
interpretable — and nothing else; the objects cannot be handed back to a stage
([files](files.md)).

## `kuiva.interface.api`

The functions with **no stage of their own**, collected here because otherwise they are
findable only by knowing the module path:

| call | what it is for |
|---|---|
| `api.scalar_x2c_reference(molecule, ...)` | the front end as a function — what [`ScalarSCF`](stages/ScalarSCF.md) wraps |
| `api.spinor_reference(molecule_or_data, ...)` | front end → working basis → spinors → factors, in one call — what [`Reference`](stages/Reference.md) wraps |
| `api.casscf(reference, ...)` / `api.casci(reference, ...)` | the CASSCF/CASCI drivers; `coeff=` takes orbitals with no stage to hang them on |
| `api.casscf_from_checkpoint(reference, path, ...)` | materialize a finished CASSCF ([clusters](../guide/clusters.md#restarting)) |
| `api.active_space_for(reference, character=..., n_active=...)` | resolve an active-space request into the object the drivers take, so it can be inspected before a run is committed to |
| `api.avas_active_space(reference, atom=..., l=...)` | the AVAS space **and its rotated orbitals**, so the projection eigenvalues can be looked at before committing to a threshold |
| `api.localize_active_space(reference, space, sites)` | the SPADE site partition ([workflows](../guide/workflows.md#which-centre-an-active-orbital-belongs-to)) |
| `api.project_to_basis(source, target, coeff, ...)` | the basis projection, returning coefficients and diagnostics without running anything; `api.projected_active_space(plan, target, n_elec)` is the target-basis space it lands on |
| `api.property_matrices(reference, source)` | `H` and the moment matrices **without writing a file**, for comparing two calculations in memory through `.analyse()`; `api.property_dump` is this plus the write |
| `api.spin_analysis(reference, source)` / `api.assign_states(reference, source)` | `<S^2>` per block and the assignment offer, for a result produced by hand (a bare `api.casci`, which has no stage to call the methods on) |
| `api.memory_plan(nao, ...)` | the phase-by-phase memory estimate the pre-flight prints, as data — a function of dimensions only, answering "will this fit?" before any array or SCF exists |
| `api.build_mole(molecule)` | the PySCF `Mole` a `Molecule` resolves to, including the decorated atom labels (`"Ti2"`) that per-atom bases and configurations are addressed by |
| `kuiva.interface.pyscf_bridge.run_scalar_aoc(element, configuration, basis=...)` | a scalar X2C SCF on one atom or ion, averaged over a configuration — spherical, one radial function per shell ([scf-reference](../methods/scf-reference.md#average-of-configuration-atoms)). ⚠ Fractional occupations; the pipeline stages are not validated on such a reference |
| `Molecule.from_xyz_file(path, basis)` / `.from_xyz_string(xyz, basis)` | molecules from XMol input ([Molecule](stages/Molecule.md)) |

The container the multireference layer starts from is `api.SpinorReference` (`.data`,
`.orth`, `.spinors`, `.factors`); its `h_one_electron()` returns the full two-component
one-electron Hamiltonian in the AO basis — the operator the correlated energy is an
expectation value of — and `.spinors_in_ao()` the guess spinors.

## The module drivers

One level further down, the drivers each stage ultimately calls — the seams a new method
plugs into:

| driver | contract |
|---|---|
| `kuiva.mcscf.orbopt.optimize_orbitals(factors, h_ao, c, spaces, ci_solver, ...)` | the shared optimizer: `ci_solver(ints) -> (E, gamma, Gamma)` is the only thing distinguishing a cheap CI, a full CI and a DMRG ([casscf](../methods/casscf.md)); `callback(info)` fires per macro-iteration, returning `False` stops the run |
| `kuiva.mcscf.events.optimize_orbitals_events(...)` | the event-gated sibling for adaptive solvers ([casscf](../methods/casscf.md#adaptive-solvers-the-optimizer-owns-the-space)) |
| `kuiva.mcscf.casci.FullCISolver` / `kuiva.mcscf.casci.casscf` | the full-CI solver and the CASSCF composition |
| `kuiva.mcscf.preopt.preoptimize` / `cheap_ci` | the selected-CI pre-optimization ([active-spaces](../methods/active-spaces.md#the-cheap-ci-pre-optimization)) |
| `kuiva.dmrg.DMRGSolver`, `kuiva.dmrg.NetworkGraph`, `kuiva.dmrg.bond_series` | the tensor-network solver, its topology owner, and the bond-dimension extrapolation series ([dmrg](../methods/dmrg.md)) |
| `kuiva.pt.nevpt2.sc_nevpt2` / `kuiva.pt.network.sc_nevpt2_dmrg` | the perturbation on either reference ([nevpt2](../methods/nevpt2.md)) |
| `kuiva.props.dump.write_dump`, `kuiva.props.pseudospin.write_pseudospin` | the formatted writers |
| `kuiva.amf.amf_correction(mole, method=...)` | the atomic mean-field correction on its own, one element at a time — `(Δh_sf, Δw)` in the AO basis, with `.report()` |
| `kuiva.util.resources.calculation(label)` / `.clear()` | scoping several calculations in one process ([configuration](../guide/configuration.md#the-memory-limit)) |
| `kuiva.util.logging.add_file_handler`, `set_verbosity` | the output stream |

## Extras: atomic Slater-Condon parameters

`kuiva.extras` holds self-contained special-purpose methods that ship with the code and are
usable, but are **not** part of the multireference pipeline and are not maintained at the
same level; they reach the core through its ordinary public interfaces, and nothing in a
calculation depends on them. There is currently one:

```python
from kuiva.extras import slater_condon_parameters

result = slater_condon_parameters("Dy", "[Xe] 4f9 5d1 6s1", basis="x2c-TZVPall-2c",
                                  shells=("4f", "5d", "6s"), file="dy_i.scp")
```

Given an element and a configuration, it converges an average-of-configuration scalar X2C
SCF — spherical, one radial function per shell — and returns the radial parameters
$F^k(a,b)$, $G^k(a,b)$ and $R^k(ab;cd)$ among the named shells
[[189]](../references.md#r189)[[85]](../references.md#r85)[[181]](../references.md#r181),
together with the one-electron spin–orbit constants $\zeta_{nl}$, in a log table and a
versioned plain-text file ([files](files.md#the-slater-condon-file)).

- ⚠ The genuine cross parameters $R^k(ab;cd)$ carry the **phase** of radial functions they
  name an odd number of times, so their sign is only meaningful against a stated
  convention: Kuiva fixes each radial function positive in its outer region
  ($P_{nl}(r) > 0$ as $r \to \infty$) and states this in every file. $F^k$, $G^k$ and
  $\zeta$ are quadratic in every radial function and never depended on it.
- ⚠ The parameters are **frozen average-of-configuration** values of one configuration in
  one basis: not self-consistent values for any particular term, no correlation, and not
  comparable across bases. A parameter set fitted to experiment is a different object, and
  Hartree–Fock-level values are known to sit above one. Every file states this.
- $\zeta$ needs the two-component operator and therefore one cached four-component atomic
  solve per element; `zeta=False` keeps the run to the SCF. The 3j symbols behind the
  angular factors are evaluated in exact rational arithmetic
  [[190]](../references.md#r190), and the hydrogenic closed forms
  [[191]](../references.md#r191) are the validation targets.
