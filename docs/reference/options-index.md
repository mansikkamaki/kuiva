# Options index

Every user-facing option, alphabetically, with where it lives and where it is documented.
Stage names link to the reference page carrying the quick table and elaboration;
"[workflows]" and method pages carry the consequences. Sub-options passed inside a dict
(`solver_options=`, `projection=`, …) are in the second table.

## Constructor options

| option | on | documented |
|---|---|---|
| `active` | `CheapCI`, `CASSCF`, `CASCI` | [CASSCF](stages/CASSCF.md#quick-reference), [CASCI](stages/CASCI.md#the-inheritance-rules) |
| `allow_unconverged_scf` | `ScalarSCF` | [ScalarSCF](stages/ScalarSCF.md#when-the-scf-will-not-converge) |
| `anomaly_picture_change` | `ScalarSCF` | [ScalarSCF](stages/ScalarSCF.md#quick-reference) |
| `atomic_reference` | `ScalarSCF` | [ScalarSCF](stages/ScalarSCF.md#quick-reference); needed by AVAS and the atomic-reference charges |
| `atoms` | `Molecule` | [Molecule](stages/Molecule.md) |
| `auxbasis` | `ScalarSCF` | [ScalarSCF](stages/ScalarSCF.md#the-two-electron-route) |
| `avas` | `CheapCI`, `CASSCF`, `CASCI` | [workflows](../guide/workflows.md#avas-when-no-single-orbital-is-the-target-shell), [active-spaces](../methods/active-spaces.md#avas-the-covalent-case) |
| `axes`, `common_axis` | `PseudospinExport` | [PseudospinExport](stages/PseudospinExport.md#quick-reference) |
| `basis` | `Molecule` | [Molecule](stages/Molecule.md#the-basis-assignment) |
| `boundary_check` | `CASSCF` (optimizer) | [CASSCF](stages/CASSCF.md#quick-reference), [workflows](../guide/workflows.md#designing-a-state-average) |
| `broken_symmetry`, `bs_min_population` | `ScalarSCF` | [workflows](../guide/workflows.md#antiferromagnetically-coupled-centres-broken-symmetry) |
| `callback` | `CASSCF` | [CASSCF](stages/CASSCF.md#quick-reference), [api](api.md#the-module-drivers) |
| `character` | `CheapCI`, `CASSCF`, `CASCI` | [workflows](../guide/workflows.md#active-spaces-beyond-the-simple-case), [active-spaces](../methods/active-spaces.md#selection-by-orbital-character) |
| `charge` | `Molecule` | [Molecule](stages/Molecule.md) |
| `checkpoint`, `restart`, `checkpoint_options` | `CASSCF`, `NEVPT2` | [clusters](../guide/clusters.md#checkpointing), [files](files.md#the-casscf-checkpoint) |
| `cholesky_tol` | `ScalarSCF`, `Reference` | [ScalarSCF](stages/ScalarSCF.md#the-two-electron-route), [integrals](../methods/integrals.md#the-default-threshold-and-what-it-does-and-does-not-bound) |
| `classes` | `NEVPT2` | [NEVPT2](stages/NEVPT2.md#quick-reference) |
| `classification` | `Molecule`, `ScalarSCF` | [Molecule](stages/Molecule.md), [symmetry](../methods/symmetry.md#the-non-abelian-layer-classification-never-adaptation) |
| `classify` | `CASSCF`, `CASCI` | [CASSCF](stages/CASSCF.md#quick-reference) |
| `coeff` | `CASCI` (and `api.casscf`) | [CASCI](stages/CASCI.md#the-inheritance-rules) |
| `comments` | `PropertyDump`, `PseudospinExport` | [PropertyDump](stages/PropertyDump.md#quick-reference) |
| `configuration` | `ScalarSCF` | [ScalarSCF](stages/ScalarSCF.md#reference-configurations-configuration) |
| `conv_energy`, `conv_grad` | `CASSCF` (optimizer), `CheapCI` | [CASSCF](stages/CASSCF.md#quick-reference) |
| `conv_tol`, `max_cycle` | `ScalarSCF` | [ScalarSCF](stages/ScalarSCF.md#quick-reference) |
| `damp`, `level_shift`, `diis`, `diis_space`, `diis_start_cycle` | `ScalarSCF` | [ScalarSCF](stages/ScalarSCF.md#when-the-scf-will-not-converge) |
| `deadline` | `CASSCF`, `NEVPT2` | [clusters](../guide/clusters.md#stopping-before-the-queue-does-deadline) |
| `decoupling_options` | `ScalarSCF` | [x2c](../methods/x2c.md#local-decoupling-dlu) |
| `deleted_virtual`, `frozen_core` | `NEVPT2` | [NEVPT2](stages/NEVPT2.md#quick-reference) |
| `dims`, `rule`, `sites` | `PseudospinExport` | [PseudospinExport](stages/PseudospinExport.md#quick-reference) |
| `environment` | `Molecule` | [Molecule](stages/Molecule.md#environmentpoint_charges-), [workflows](../guide/workflows.md#embedding-in-a-crystal-point-charges) |
| `factors` | `ScalarSCF` | [ScalarSCF](stages/ScalarSCF.md#the-two-electron-route) |
| `fitting` | `ScalarSCF` | [ScalarSCF](stages/ScalarSCF.md#the-two-electron-route), [integrals](../methods/integrals.md#three-routes-to-the-decomposition-one-result) |
| `fock` | `NEVPT2` | [nevpt2](../methods/nevpt2.md#the-zeroth-order-hamiltonian) |
| `g_electron` | `PseudospinExport` | [PseudospinExport](stages/PseudospinExport.md#quick-reference) |
| `gauge_origin` | `ScalarSCF` | [ScalarSCF](stages/ScalarSCF.md#the-gauge-origin) |
| `graph` | `CASSCF` (`solver="dmrg"`) | [CASSCF](stages/CASSCF.md#solverdmrg-the-tree-tensor-network) |
| `guess_from` | `ScalarSCF` | [workflows](../guide/workflows.md#an-scf-from-another-scf-guess_from) |
| `imaginary_shift`, `shift` | `NEVPT2` | [NEVPT2](stages/NEVPT2.md#quick-reference) |
| `include_dipole`, `include_l_s` | `PropertyDump` | [PropertyDump](stages/PropertyDump.md#quick-reference) |
| `inactive_tol` | `PropertyDump` | [PropertyDump](stages/PropertyDump.md#quick-reference) |
| `init_guess` | `ScalarSCF` | [ScalarSCF](stages/ScalarSCF.md#when-the-scf-will-not-converge) |
| `kramers_rotation` | `CASSCF` (optimizer) | [casscf](../methods/casscf.md#keeping-the-orbitals-kramers-paired) |
| `kramers_stability` | `CASSCF` (optimizer) | [casscf](../methods/casscf.md#releasing-the-constraint-is-the-symmetric-solution-a-minimum) |
| `manifold_options` | `PseudospinExport` | [PseudospinExport](stages/PseudospinExport.md#quick-reference) |
| `max_bond` | `PseudospinExport` (and `solver_options`) | [CASSCF](stages/CASSCF.md#solverdmrg-the-tree-tensor-network) |
| `max_iter` | `CASSCF` (optimizer), `CheapCI` | [CASSCF](stages/CASSCF.md#quick-reference) |
| `max_step`, `memory`, `active_active` | `CASSCF` (optimizer) | [CASSCF](stages/CASSCF.md#quick-reference) |
| `memory_gb` | `ScalarSCF` | [configuration](../guide/configuration.md#the-memory-limit) |
| `method` | `ScalarSCF` | [ScalarSCF](stages/ScalarSCF.md#the-hamiltonian-by-name) |
| `mode` | `CASSCF` (optimizer), `CheapCI` | [casscf](../methods/casscf.md#the-step-and-the-three-ways-to-take-it) |
| `n_active`, `n_active_elec` | `CheapCI`, `CASSCF`, `CASCI`; planning on `ScalarSCF` | [CASSCF](stages/CASSCF.md#quick-reference) |
| `n_states` | `CheapCI`, `CASSCF`, `CASCI`; planning on `ScalarSCF` | [workflows](../guide/workflows.md#designing-a-state-average), [CASSCF](stages/CASSCF.md#per-irrep-selection) |
| `nevpt2` (planning flag) | `ScalarSCF` | [ScalarSCF](stages/ScalarSCF.md#quick-reference) |
| `norm_cutoff`, `degeneracy_tol`, `on_split` | `NEVPT2` (also `solver_options`) | [NEVPT2](stages/NEVPT2.md#quick-reference) |
| `nuclear_model` | `Molecule` | [x2c](../methods/x2c.md#the-nuclear-charge-model) |
| `orbit_pivots`, `one_centre` | `ScalarSCF`, `Reference` | [integrals](../methods/integrals.md#one-centre-pivoting-symmetry-enforced-structurally) |
| `point_group` | `Molecule`, `ScalarSCF` | [symmetry](../methods/symmetry.md) |
| `preserve_symmetry` | `CASSCF` | [CASSCF](stages/CASSCF.md#per-irrep-selection) |
| `project_from`, `projection` | `CASSCF` | [workflows](../guide/workflows.md#a-casscf-from-a-different-basis-set-project_from) |
| `property_picture_change` | `ScalarSCF` | [limitations](../limitations.md#property-operators), [properties](../methods/properties.md#the-operator-matrices) |
| `reference` | `ScalarSCF` | [ScalarSCF](stages/ScalarSCF.md#quick-reference) |
| `report` | most stages | the standard output blocks; `False` silences them |
| `rotate_frame` | `PseudospinExport` | [PseudospinExport](stages/PseudospinExport.md#quick-reference) |
| `scheme`, `threshold` (orthogonalization) | `Reference` | [Reference](stages/Reference.md#quick-reference) |
| `screening`, `screening_options` | `ScalarSCF` | [soc](../methods/soc.md), [ScalarSCF](stages/ScalarSCF.md#the-hamiltonian-by-name) |
| `second_order`, `stability` | `ScalarSCF` | [ScalarSCF](stages/ScalarSCF.md#is-the-converged-solution-a-minimum-at-all) |
| `seed` | `PseudospinExport` (and `solver_options`) | [PseudospinExport](stages/PseudospinExport.md#quick-reference) |
| `signals` | `CASSCF`, `NEVPT2` | [clusters](../guide/clusters.md#the-kill-nobody-announced-signals) |
| `solver`, `solver_options` | `CASSCF` (`solver_options` also `CASCI`) | [CASSCF](stages/CASSCF.md#quick-reference) |
| `spin` | `Molecule` | [Molecule](stages/Molecule.md) |
| `threshold` (character selection) | `CheapCI`, `CASSCF`, `CASCI` | [active-spaces](../methods/active-spaces.md#selection-by-orbital-character) |
| `title` | `PropertyDump`, `PseudospinExport` | [PropertyDump](stages/PropertyDump.md#quick-reference) |
| `unit` | `Molecule`, `Environment` | [Molecule](stages/Molecule.md), [notation](../notation.md#units-and-physical-constants) |
| `weights` | `CASSCF`, `CASCI` | [CASSCF](stages/CASSCF.md#quick-reference) |
| `with_soc` | `ScalarSCF` | [workflows](../guide/workflows.md#turning-spin-orbit-coupling-off) |

## Dict sub-options

| sub-option | inside | on | documented |
|---|---|---|---|
| `adaptive`, `policy`, `propose_sweeps` | `solver_options` | `CASSCF("dmrg")` | [CASSCF](stages/CASSCF.md#solverdmrg-the-tree-tensor-network) |
| `backend`, `block`, `max_subspace` | `solver_options` | `CASSCF("ci")`, `CASCI` | [CASSCF](stages/CASSCF.md#solverci-the-conventional-ci) |
| `bond_schedule`, `bond_steps`, `expansion`, `expansion_sweeps` | `solver_options` | `CASSCF("dmrg")` | [CASSCF](stages/CASSCF.md#solverdmrg-the-tree-tensor-network) |
| `carry`, `scheme`, `repair_pairing` | `projection` | `CASSCF` | [workflows](../guide/workflows.md#a-casscf-from-a-different-basis-set-project_from), [integrals](../methods/integrals.md#carrying-orbitals-between-basis-sets) |
| `enforce_kramers`, `kramers`, `warm_start` | `solver_options` | `CASSCF("ci")`, `CASCI` | [ci](../methods/ci.md#the-kramers-restricted-mode) |
| `interaction`, `backend`, `uncontract` | `screening_options` | `ScalarSCF` | [soc](../methods/soc.md#the-interaction-coulomb-gaunt-breit) |
| `max_bond`, `max_sweeps`, `conv_tol`, `davidson_tol`, `trunc_tol` | `solver_options` | `CASSCF("dmrg")` | [CASSCF](stages/CASSCF.md#solverdmrg-the-tree-tensor-network) |
| `max_pairs`, `n_shells`, `threshold` | `avas` | `CheapCI`, `CASSCF`, `CASCI` | [workflows](../guide/workflows.md#avas-when-no-single-orbital-is-the-target-shell) |
| `min_interval` | `checkpoint_options` | `NEVPT2` | [NEVPT2](stages/NEVPT2.md#quick-reference) |
| `n_roots`, `max_roots`, `max_outer` | `manifold_options` | `PseudospinExport` | [PseudospinExport](stages/PseudospinExport.md#quick-reference) |
| `partition`, `source` | `decoupling_options` | `ScalarSCF` | [x2c](../methods/x2c.md#local-decoupling-dlu) |
| `symmetry`, `sector` | `solver_options` | `CASSCF("dmrg")` | [CASSCF](stages/CASSCF.md#per-irrep-selection) |
| `tau`, `event_interval`, `max_event_interval`, `trust_floor`, `keep_memory_on_adopt` | optimizer keywords | `CASSCF` (event driver) | [CASSCF](stages/CASSCF.md#quick-reference) |
