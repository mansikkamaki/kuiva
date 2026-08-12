# Kuiva validation suite

The default suite is **laptop-fast** and needs nothing beyond Kuiva's own dependencies; the
external programs (OpenMolcas, DIRAC) are only required to *regenerate* reference data, which
is committed.

```bash
source setup.sh
pytest                     # default suite: fast tests only (~8 min)
pytest -m slow             # the slow tests (heavy-atom four-component solves; hours)
pytest -m ''               # everything — budget >= 4 h and pipe through `tee`, never `tail`

# The quantum-computing tests run under a DIFFERENT interpreter: Qiskit needs >= 3.10, so
# external/venv_qc is its own venv from its own python. They are deselected from every
# command above, and the repository root goes on PYTHONPATH so they test the working tree.
PYTHONPATH=$PWD external/venv_qc/bin/python -m pytest -m qc

# The rest of the kuiva/qc/ tests need no framework and run in the default suite above.
# Under venv_qc they also check that the classical layers work on Python 3.10 + NumPy 2.x.
PYTHONPATH=$PWD external/venv_qc/bin/python -m pytest tests/test_qc_*.py -m ''
```

Long runs can replay recorded intermediate stages instead of recomputing them — see
[Stage checkpoints](#stage-checkpoints) below. It is **off by default**.

⚠ The development machine thermally clamps under sustained load, so **wall time is not a cost
measure here**: judge cost from `cpu_seconds` (every generator records both, plus the injected
idle, via `tests/generate/thermal.py`), and keep wall-clock timeouts generous — a run being
cooled looks exactly like a run that is too big.

## The tiers

* **Tier 0 — analytic / internal consistency.** Exactly solvable models, hydrogenic spinors
  with known SOC splittings, invariance checks (energy under active-orbital rotation and
  global phase, RDM hermiticity and traces, `T² = −1` on odd-electron states). Cheap, catches
  most bugs.
* **Tier 1 — cross-validation against PySCF itself**, tiny systems, minimal bases. The test
  is *correctness*, not accuracy.
* **Tier 2 — external cross-checks (OpenMolcas and DIRAC).** Run **once**, parsed into
  reference files committed under `tests/reference/`; thereafter the suite compares against
  the stored files and never re-runs the external code. The committed data — never a table in
  prose — is the authority for any number.
* **Tier 3 — tensor-network structure on polynuclear systems** for which no program can
  produce a reference. Assertions are restricted to theorems and counting (Clebsch–Gordan
  decompositions, Kramers parity, Lieb–Mattis, graph invariants) and, for three systems, the
  experimentally established ground spin.

⚠ A check whose two sides share an implementation cannot see an error in that implementation,
however many digits it agrees to. Weigh a new check by what it can *fail on*, not by how
tightly it agrees. Every test carries an explicit tolerance and a note on the physically
meaningful one.

## Layout

| Path | Role |
|---|---|
| `tests/generate/systems.py` | **One source of truth** for the test systems: geometries, charges, active spaces, state counts. Both tiers import it, so a Tier-2 system always has a scalar Tier-1 counterpart under the same key. |
| `tests/generate/tier3_systems.py` | The same, for the Tier-3 tensor-network systems: paramagnetic centres, exchange graphs, the effective-spin-model ED. |
| `tests/generate/*.py` | One generation script per committed reference (`tier1_pyscf`, `tier2_molcas`, `tier2_dirac`, `tier2_kuiva`, `x2camf_*`, `amf_*`, `nevpt2_*`, `tier3_invariants`, `crosscheck_external_basis`) plus benchmark/characterization scripts that write `temp/` rather than a reference (`bench_*`, `dlu_accuracy`, `cholesky_threshold`, `amf_open_shell_cost`, `spec_gaps_*`). Each script's docstring states what it produces, its cost, and its pinned comparison variables. |
| `tests/generate/progress.py` | Heartbeat reporting for long runs. Runs report on **themselves**; the checker uses the heartbeat file plus `/proc/<pid>` — never a process command line, which matches watcher shells as readily as the job. |
| `tests/reference/*.json` | Committed reference data — the authority for every number. Regenerate only when a method changes fundamentally. |
| `tests/stages.py` | Disk-stored intermediate stage checkpoints for the heavy tests, and the CLI that lists/purges them. |
| `tests/fockspace.py` | A brute-force second-quantization reference for SC-NEVPT2: dense ladder operators over a whole Fock space, sharing **no code** with `kuiva.pt`/`kuiva.rdm`/`kuiva.ci`. |
| `tests/test_*.py` | The suite. Each module's docstring states what it asserts, against what, and why that comparison can fail — read it before adding to the module. |

## The systems

Every Tier-2 system has a scalar Tier-1 counterpart under the same key, which is what lets the
suite separate *"is the scalar CASSCF right?"* from *"is the SOC treatment right?"*.

⚠ The two counterparts do not share a state count, and `System` carries both: `soc_states`
bounds the two-component spectrum's `2J+1` manifolds, `scalar_states` the spin-free `2S+1`
ones. A count that is a boundary of one generally cuts a block of the other, and a cut block
breaks the state average self-consistently. Where PySCF's spin-adapted average and Kuiva's
lowest-roots average would not span the same set, `scalar_nroots` names a smaller complete
ensemble and `tier1_pyscf.json` carries a separate `scalar_crosscheck` record.

| Key | System | What it catches | Tier 2 |
|---|---|---|---|
| `ne` | Ne | cheapest end-to-end smoke test | — |
| `zn2p` | Zn²⁺ (3d¹⁰) | CASCI over a *full* active space must equal SCF **exactly** | — |
| `hi` | HI | light–heavy bond, molecular σ/σ* CAS | — |
| `bi` | Bi (6p³) | multireference **and** strongly SOC-coupled; 20-state p³ manifold | Molcas, DIRAC |
| `tlh` | TlH | molecular heavy diatomic, SOC-split Π states | Molcas |
| `ce3p` | Ce³⁺ (4f¹) | minimal f-element: 6+8 structure, analytic g = 6/7 | Molcas, DIRAC |
| `yb3p` | Yb³⁺ (4f¹³) | the one-**hole** counterpart; *inverted* 8+6 multiplet, g = 8/7 | Molcas, DIRAC |
| `dy3p` | Dy³⁺ (4f⁹) | strongly correlated f shell, the archetypal SMM ion; ⁶H₁₅/₂, g = 4/3; averages over **134** spinor roots with SOC on and **66** with it off — the two spectra have different manifolds | Molcas |
| `cecl3` | CeCl₃ | single f¹ site **in a ligand field**: anisotropic g | Molcas |
| `ticl3` | TiCl₃ | single d¹ site in a ligand field; the monomer the dimers invert back to | Molcas |
| `ti2cl6` | Ti₂Cl₆ | **multi-site**: two coupled d¹ centres, 100-state d¹⊗d¹ manifold | Molcas |
| `ti2cl6_far` | Ti₂Cl₆ at 25 Å | **additivity**: the spectrum must factorise into local multiplets | Molcas |

Ce/Yb are deliberately a particle–hole pair (1 electron vs 1 hole in the same 14 spinors) —
that symmetry is what CI string-addressing bugs tend to break. The multi-site pair is d¹
rather than f¹ so its references regenerate in CPU-minutes rather than CPU-hours; `ti2cl6_far`
is a **singlet diradical** whose closed-shell SCF needs the second-order fallback (records
carry `scf_second_order`).

### Tier-3 systems

| Key | System | Topology | Network it exercises |
|---|---|---|---|
| `mn3_linear` | Mn(II)₃ chain | path P3 | plain MPS — the baseline |
| `dy2_n2rad` | Dy₂ + N₂³⁻ radical bridge | path P3, local dims (16, 2, 16) | MPS with **inhomogeneous** local dimensions |
| `fe3_oxo` | [Fe₃(μ₃-O)] basic carboxylate | cycle C3 | minimal **frustrated** loop |
| `fe4_star` | Fe₄ star SMM | star K(1,3) | **TTNS** — the minimal genuine tree |
| `mn4ca_oec` | Photosystem II Mn₄CaO₅ | triangle + pendant | **hierarchical**: loop with a tree branch |
| `fe4s4` | [4Fe-4S] ferredoxin cubane | complete graph K4 | PEPS-like; no ordering is short-ranged |
| `cr8_ring` | Cr₈ AF ring | cycle C8 | periodic MPS / long-range bond |
| `cr7ni_ring` | Cr₇Ni ring | cycle C8, one defect | same topology, different local quantum numbers |

All are real structural motifs, truncated only where the truncation cannot change the magnetic
topology; truncation never adds or removes a paramagnetic centre or an exchange pathway. Three
carry experimental ground spins (`fe4_star` S = 5, `cr8_ring` S = 0, `cr7ni_ring` S = 1/2),
which agree with Lieb–Mattis where it applies; the theorem helper returns `None` rather than a
number when its hypotheses fail, and is cross-checked against dense ED of the effective spin
model. A test asserts every Tier-3 system stays beyond the conventional-CI ceiling — if one
ever becomes cheap enough to compute, it belongs in Tier 1/2 with a real reference.

## Cross-code comparison

Kuiva, OpenMolcas and DIRAC will **not** agree to many digits, and the suite is built so this
is never mistaken for an error: the Hamiltonians differ (X2C with SOC at the CI step; DKH2 +
AMFI; 4c/X2C), the orbital treatments differ, and phases are arbitrary on all sides — so
everything is compared through **phase-invariant** quantities (`kuiva.props.multiplet`):
degeneracy patterns, relative energies, and the moment invariant `M_ij = Tr_block(μ_i μ_j)`
with its principal g values. Raw matrix elements are never compared. Tier 1 runs both in the
project default basis and in the external code's own basis, so a Tier-1/Tier-2 discrepancy is
attributable to the Hamiltonian and method, not the basis.

| Quantity | Tolerance | Why |
|---|---|---|
| Scalar X2C SCF (Kuiva vs stored) | 1e-8 Eh | same call; pure regression lock |
| CASSCF energy | 1e-7 Eh | variational optimum, converged to 1e-9 |
| Kuiva CASCI vs scalar CASSCF | 1e-7 Eh | *identical* problem in a doubled basis — SOC off |
| Degeneracy pattern | **exact** | fixed by symmetry; no method can change it |
| Landé g of a free ion | 1% | analytic; independent of every code (observed: 0.04%) |
| g isotropy of a free ion | 1e-3 | spherical symmetry, internal consistency — **only for an average over a whole term**: one complete `2J+1` manifold is a boundary but not an invariant ensemble, and the anisotropy left by a sub-manifold average is set by rounding, not by the method (measured 60× apart between two BLAS libraries on identical input) |
| Cross-code SOC splittings | 15% | genuine method spread between the *external* codes; it does not measure Kuiva, whose own bands are separate and tighter |
| Kuiva 4c vs DIRAC 4c, same atom | 1e-5 Eh | same method, different program |
| X2C+AMF vs DIRAC 4c j-splitting | 0.5% | the meaningful figure for a picture-change treatment is a fraction of a percent |
| X2C+AMF vs 1e-X2C, occupied spectrum | ≥100× better | a statement about the *correction*; an absolute band could be met by a basis coincidence |
| vs experiment | 30% | anchor against gross error, **not** an accuracy claim |
| Tensor-product additivity at 25 Å | 1 cm⁻¹ | exact by locality, within one code |
| SC-NEVPT2 degeneracy | 5e-4 Eh | PySCF's degenerate CASCI roots mix arbitrarily |

A free ion's `2J+1` multiplets must be degenerate to a *physically* reasonable accuracy —
0.1 cm⁻¹ already implies different physics — so that is asserted as a physical requirement,
not recorded as a tolerance. A band set at the observed spread is a band that fails on the
next system; a ligand-field band needs an absolute floor as well as a percentage.

## Stage checkpoints

The full suite is hours, most of it four-component atomic solves that the change under test
usually cannot have moved. `tests/stages.py` records a calculation stage's output on disk so
a later run can replay it:

```bash
pytest -m '' --checkpoints=refresh   # produce the checkpoint data (compute, overwrite)
pytest -m '' --checkpoints=on        # replay what is valid; a test's own subject still runs
                                     # (also the second pass that fills a refresh run's gaps)
pytest -m '' --checkpoints=trust     # replay everything valid, INCLUDING subjects
python tests/stages.py               # what is stored, and what is still valid
python tests/stages.py --stages      # the registry: each stage's modules and source closure
python tests/stages.py --purge      # clear it
```

Nothing is on unless asked for: a plain `pytest`, and a fresh clone, compute everything.
`tests/checkpoints/` is git-ignored and is **never** a reference. Two rules decide every case:

1. **Staleness is automatic** — each stage fingerprints the transitive import closure of its
   declared modules (docstrings and comments stripped). The fingerprint does **not** cover
   basis-set data, MKL or the machine, so a checkpoint is a replay, never a claim of
   portability.
2. **A test is never served a checkpoint of the stage it is testing.** The subject —
   declared with `@pytest.mark.stage_under_test(...)` — and everything downstream are
   computed; upstream stages may be replayed. ⚠ Any test asserting a solve or hit count must
   carry the mark, or a warm cache underneath makes it pass vacuously;
   `test_stages.py` enforces this from the sources.

`--checkpoints=trust` replays subjects too, downgrading those tests to "the recorded run
still matches the reference". Use it while iterating; never report a green `trust` run as
validation. When checkpoints are on, the persistent X2CAMF correction cache is pointed at a
fingerprinted directory under `tests/checkpoints/` and switched off for any test whose
subject it would hide; a `refresh` pass leaves a gap by construction, so follow it with an
`on` pass.

Adding a stage is a claim that its output is reproducible from its declared modules and its
key alone; the payload is a plain mapping of arrays and scalars, never a live object. When in
doubt, leave it out.

## Regenerating

Reference generation runs **once**; the fast suite never invokes OpenMolcas or DIRAC.

```bash
source setup.sh
python tests/generate/tier1_pyscf.py --fast-only     # ~2 min
python tests/generate/tier2_molcas.py                # minutes per system
python tests/generate/tier2_dirac.py --memory-gb 6   # ~20 s per system
python tests/generate/x2camf_dirac.py                # 1 s (Ne) … 56 s (Xe) per record
python tests/generate/x2camf_plugin.py               # ~2 min for Ne+Ar, both bases
python tests/generate/x2camf_molcas_amfi.py          # ~30 s (Ne) … 70 s (Ar) per ion
```

Generators write each record as soon as it exists and take `--only`/`--timeout`/`--max-wall`
style bounds, so a run stopped at its budget still leaves what it finished. ⚠ Do not assume
the external program is the slow half: the uncontracted four-component atomic solve on the
*Kuiva* side grows as roughly `n4c⁴` and dominates for heavy atoms.

### External-code setup notes

Hard-won and easy to re-lose:

* **OpenMolcas** — the magnetic-moment matrices must be read from `$Project.aniso`, which
  `&SINGLE_ANISO` writes. The obvious-looking `rassi.h5` datasets `SOS_ANGMOM_*` / `SOS_SPIN_*`
  exist but are written as **all zeros** unless extra RASSI keywords are given, so taking them
  silently yields a reference with vanishing magnetic moments.
* **The `x2camf` plugin** — built by `scripts/bootstrap/80_x2camf.sh` and optional. Three
  things bite. Its `install_requires=["numpy"]` is unpinned and pip resolves it to **NumPy
  2.x**, upgrading the venv off the pinned baseline and killing PySCF with `numpy.dtype size
  changed`; install with `--no-deps`. Its atomic solver **segfaults on the default 8 MB
  stack** — Ne is fine, Ar dumps core — so `RLIMIT_STACK` is raised inside
  `kuiva.amf.x2camf_plugin` rather than in a shell wrapper. And it reports failure by
  **printing to stdout and returning anyway**, or by calling **`exit(99)`**, which kills the
  Python process with no exception — the first is captured and turned into a `WARNING`, the
  second is why the generator writes every record as soon as it exists.
* **OpenMolcas, for the atomic AMFI records** — nothing asks for DKH2 or AMFI by keyword;
  OpenMolcas switches both on for an ANO-RCC basis, and `parse_output` checks the two output
  lines that say so rather than assuming it. `MOLCAS_WORKDIR` must exist **before** `pymolcas`
  runs, and the sandbox environment must be sourced or `gateway.exe` fails to find `libimf.so`.
* **DIRAC** — `.CIROOTS` in linear symmetry takes `2Ω` with a `g`/`u` label, and roots must be
  requested per Ω according to how many J levels contain that Ω (Bi p³: 5/4/1, not 4/4/2).
  `.CI PROGRAM`'s value must be flush-left. `**MOLTRA.ACTIVE` must span the inactive core as
  well as the active shell. `.CLOSED SHELL` counts must follow the *configuration* — DIRAC
  fills the lowest spinors per irrep without checking, so a wrong gerade/ungerade split
  silently leaves a shell short. Pass `--gb` but **not** `--ag`: capping the dynamic pool makes
  KRCI abort. AMFI's atomic SCF needs `.MXITER`/`.AMFICH` for some open-shell atoms.
* **DIRAC, for an open-shell (average-of-configuration) atom** — `*SCF.OPEN SHELL` takes
  `1` then `q/n_g,n_u`, i.e. counts **per fermion irrep**, and ⚠ **parity is `(-1)^l`: `s` and
  `d` are gerade, `p` *and* `f` are ungerade.** Putting a 4f shell in the gerade column is a
  one-character error that converges cleanly and reports a "4f splitting" of 116076 cm⁻¹ where
  the answer is 2006. The parser therefore checks the angular-momentum label DIRAC prints on
  each block, and takes the valence manifold from the **fractionally occupied** set rather
  than by energy index — "the top `4l+2` occupied spinors" is false for a lanthanide, where 4f
  lies below 5s and 5p. `--reparse` re-reads existing outputs, because a lanthanide run is
  5–7 minutes.
* **DIRAC, for a controlled comparison against PySCF** — three defaults differ and each would
  show up as method error. `**INTEGRALS.NUCMOD 1` selects a **point nucleus** (DIRAC defaults
  to a Gaussian charge distribution, PySCF to a point); `**GENERAL.CVALUE` sets the speed of
  light (DIRAC 26.1 ships CODATA 2022, PySCF CODATA 2018); `**INTEGRALS *READIN.UNCONTRACTED`
  decontracts, matching Kuiva's `uncontract=True`. `**HAMILTONIAN.DOSSSS` computes the
  `(SS|SS)` integral class explicitly — without it DIRAC substitutes the Visscher point-charge
  correction while PySCF's `dhf.DHF` computes it. The nuclear model is the one that really
  matters; it is a genuine physical effect that grows with Z.

---

## References for the validation tooling

The user-facing `README.md` carries the citations for what **`kuiva` itself implements**; the
ones below belong to software and theorems used *only* to produce or assert reference data,
and they live here rather than there.

### Reference-data programs (not `kuiva` dependencies)

- **OpenMolcas** (v26.06) — RASSI spin–orbit energies and magnetic-moment matrices; DKH2 + AMFI
  for the method-decomposition table. F. Aquilante _et al._, _J. Chem. Phys._ **152**, 214117
  (2020), DOI:10.1063/5.0004835; G. Li Manni _et al._, _J. Chem. Theory Comput._ **19**,
  6933–6991 (2023), DOI:10.1021/acs.jctc.3c00182. Spin–orbit state interaction:
  P.-Å. Malmqvist, B. O. Roos, B. Schimmelpfennig, _Chem. Phys. Lett._ **357**, 230–240 (2002),
  DOI:10.1016/S0009-2614(02)00498-0. Douglas–Kroll–Hess: M. Douglas, N. M. Kroll, _Ann. Phys._
  **82**, 89 (1974), DOI:10.1016/0003-4916(74)90333-9; B. A. Hess, _Phys. Rev. A_ **33**, 3742
  (1986), DOI:10.1103/PhysRevA.33.3742. AMFI: B. A. Hess, C. M. Marian, U. Wahlgren, O. Gropen,
  _Chem. Phys. Lett._ **251**, 365–371 (1996), DOI:10.1016/0009-2614(96)00119-4. Cholesky
  decomposition of the ERIs (`RICD`, used to keep the runs affordable): F. Aquilante,
  P.-Å. Malmqvist, T. B. Pedersen, A. Ghosh, B. O. Roos, _J. Chem. Theory Comput._ **4**, 694–702
  (2008), DOI:10.1021/ct700263h. `SINGLE_ANISO` pseudospin/g-tensor extraction:
  L. F. Chibotaru, L. Ungur, _J. Chem. Phys._ **137**, 064112 (2012), DOI:10.1063/1.4739763.
- **DIRAC** (26.1) — four-component atomic references and 2c KRCI spin–orbit spectra. DIRAC
  (2026), written by H. J. Aa. Jensen, R. Bast, A. S. P. Gomes, T. Saue, L. Visscher _et al._,
  https://www.diracprogram.org; T. Saue _et al._, _J. Chem. Phys._ **152**, 204104 (2020),
  DOI:10.1063/5.0004844; DIRAC26, DOI:10.5281/zenodo.3572669. Its X2C: M. Iliaš, T. Saue,
  _J. Chem. Phys._ **126**, 064102 (2007), DOI:10.1063/1.2436882. KRCI/LUCIAREL: S. Knecht,
  H. J. Aa. Jensen, T. Fleig, _J. Chem. Phys._ **132**, 014108 (2010), DOI:10.1063/1.3276157;
  T. Fleig, J. Olsen, L. Visscher, _J. Chem. Phys._ **119**, 2963 (2003), DOI:10.1063/1.1590636.
  The `(SS|SS)` point-charge correction that `.DOSSSS` disables: L. Visscher, _Theor. Chem. Acc._
  **98**, 68 (1997), DOI:10.1007/s002140050280; the Gaunt term: J. A. Gaunt, _Proc. R. Soc. Lond.
  A_ **122**, 513 (1929), DOI:10.1098/rspa.1929.0037. The atomic-run protocol follows DIRAC's own
  (e)amfX2C implementation: S. Knecht, M. Repisky, H. J. Aa. Jensen, T. Saue, _J. Chem. Phys._
  **157**, 114106 (2022), DOI:10.1063/5.0095112.
- **The `x2camf` plugin** — the X2CAMF authors' own implementation, used for the term-by-term
  comparison and by the never-default `method="x2camf-external"`. C. Zhang, L. Cheng, _J. Phys.
  Chem. A_ **126**, 4537 (2022), DOI:10.1021/acs.jpca.2c02181; built on pybind11, W. Jakob,
  J. Rhinelander, D. Moldovan (2017), https://github.com/pybind/pybind11.
- **Global Arrays** (5.8.2) — build dependency of MPI-parallel OpenMolcas; used by neither
  `kuiva` nor DIRAC. J. Nieplocha, R. J. Harrison, R. J. Littlefield, _Proc. Supercomputing '94_,
  340–349 (1994), DOI:10.1109/SUPERC.1994.344297; J. Nieplocha _et al._, _Int. J. High Perform.
  Comput. Appl._ **20**, 203–231 (2006), DOI:10.1177/1094342006064503.
- **Build toolchain** — Intel oneAPI 2023.2 (`icx`/`icpx`/`ifx`, MKL, Intel MPI, TBB), Intel
  Corporation; OpenMPI 5.0.10 as a portability reference, E. Gabriel _et al._, _Proc. 11th
  European PVM/MPI Users' Group Meeting_, LNCS **3241**, 97–104 (2004),
  DOI:10.1007/978-3-540-30218-6_19.

### Experimental data used as anchors

- **Atomic energy levels.** A. Kramida, Yu. Ralchenko, J. Reader and NIST ASD Team, _NIST Atomic
  Spectra Database_ (v5.11), National Institute of Standards and Technology,
  https://physics.nist.gov/asd.
- **Diatomic equilibrium geometries** (HI, TlH). K. P. Huber, G. Herzberg, _Molecular Spectra and
  Molecular Structure IV: Constants of Diatomic Molecules_, Van Nostrand Reinhold (1979).

### Theorems and molecules behind the Tier-3 suite

Tier 3 rests on theorems and on experimentally established ground spins instead of a reference
calculation. Each system is also cited at the point of use in `tests/generate/tier3_systems.py`.

- **Lieb–Mattis theorem** — exact ground-state total spin of a bipartite Heisenberg
  antiferromagnet. E. Lieb, D. Mattis, _J. Math. Phys._ **3**, 749–751 (1962),
  DOI:10.1063/1.1724276.
- **Kramers degeneracy.** H. A. Kramers, _Proc. Amsterdam Acad._ **33**, 959 (1930).
- **Fe₄ star SMM** (`fe4_star`), experimental S = 5 — [Fe₄(OMe)₆(dpm)₆]. A.-L. Barra _et al._,
  _J. Am. Chem. Soc._ **121**, 5302–5310 (1999), DOI:10.1021/ja9818755; A. Cornia _et al._,
  _Angew. Chem. Int. Ed._ **43**, 1136–1139 (2004).
- **Cr₈ ring** (`cr8_ring`), experimental S = 0 — [Cr₈F₈(O₂C^tBu)₁₆]. J. van Slageren _et al._,
  _Chem. Eur. J._ **8**, 277–285 (2002).
- **Cr₇Ni ring** (`cr7ni_ring`), experimental S = 1/2. S. Larsen _et al._, _Phys. Rev. Lett._
  **91**, 067201 (2003), DOI:10.1103/PhysRevLett.91.067201; G. A. Timco _et al._, _Nat.
  Nanotechnol._ **4**, 173–178 (2009), DOI:10.1038/nnano.2008.404.
- **Basic iron(III) carboxylate triangle** (`fe3_oxo`). R. D. Cannon, R. P. White, _Prog. Inorg.
  Chem._ **36**, 195–298 (1988), DOI:10.1002/9780470166376.ch3.
- **[4Fe-4S] ferredoxin cluster** (`fe4s4`). H. Beinert, R. H. Holm, E. Münck, _Science_ **277**,
  653–659 (1997), DOI:10.1126/science.277.5326.653. As a DMRG benchmark: S. Sharma,
  K. Sivalingam, F. Neese, G. K.-L. Chan, _Nat. Chem._ **6**, 927–933 (2014),
  DOI:10.1038/nchem.2041; Z. Li, S. Guo, Q. Sun, G. K.-L. Chan, _Nat. Chem._ **11**, 1026–1033
  (2019), DOI:10.1038/s41557-019-0337-3.
- **Photosystem II Mn₄CaO₅ cluster** (`mn4ca_oec`). Y. Umena, K. Kawakami, J.-R. Shen, N. Kamiya,
  _Nature_ **473**, 55–60 (2011), DOI:10.1038/nature09913.
- **N₂³⁻ radical-bridged Dy₂ SMM** (`dy2_n2rad`). J. D. Rinehart, M. Fang, W. J. Evans,
  J. R. Long, _J. Am. Chem. Soc._ **133**, 14236–14239 (2011), DOI:10.1021/ja206286h;
  _Nat. Chem._ **3**, 538–542 (2011), DOI:10.1038/nchem.1063.
